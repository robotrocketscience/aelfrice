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
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Final, Literal, TypeVar, cast

from aelfrice import launcher
from aelfrice import setup as _setup
from aelfrice.setup import (
    PROJECT_SETTINGS_RELPATH,
    read_settings,
    write_settings,
)

if TYPE_CHECKING:
    from aelfrice.lock_gaps import LockGapReport
    from aelfrice.models import Belief
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
# Unapplied typed locks (#1622)
# ---------------------------------------------------------------------------


def _lock_gaps_on_windows() -> bool:
    """`lock_gaps._on_windows`, looked up at call time so a test can patch it."""
    from aelfrice import lock_gaps  # noqa: PLC0415

    return lock_gaps._on_windows()  # pyright: ignore[reportPrivateUsage]


def diagnose_lock_gaps(
    store_path: str, project_root: Path | None = None,
) -> "LockGapReport":
    """Return the typed `/aelf:lock` requests still unapplied (#1622).

    The `[hook_audit]` switch is resolved from `project_root` so a
    project-local `.aelfrice.toml` that disables the audit makes the
    answer unknown rather than zero. Read-only and fail-soft: see
    `aelfrice.lock_gaps.detect_lock_gaps`.
    """
    import io  # noqa: PLC0415

    from aelfrice.hook_audit import load_hook_audit_config  # noqa: PLC0415
    from aelfrice.lock_gaps import detect_lock_gaps  # noqa: PLC0415

    # Config warnings belong to the hook's stderr, not to doctor's report.
    cfg = load_hook_audit_config(project_root, stderr=io.StringIO())
    return detect_lock_gaps(store_path, audit_enabled=cfg.enabled)


# ---------------------------------------------------------------------------
# Dangling-edge check (#1375)
# ---------------------------------------------------------------------------

# How many per-type rows the dangling-edge section prints before it
# stops. The count is the finding; the type breakdown is only there to
# point at which producer to look at, and a long tail of one-row types
# would bury the total.
DANGLING_EDGE_TYPES_SHOWN: Final[int] = 5


@dataclass(frozen=True)
class DanglingEdgeStats:
    """Counts of `edges` rows whose endpoints are absent from `beliefs` (#1375).

    The `edges` table has `PRIMARY KEY (src, dst, type)` and no foreign
    key, and `insert_edge` gates only on federation ownership —
    `assert_local_ownership` is a documented no-op for an id the store
    has never seen. So an edge naming a belief that does not exist is
    accepted silently and nothing has ever reported it.

    `total` counts *edges*: an edge missing both endpoints is one row,
    counted once. `missing_src` and `missing_dst` count endpoints, so
    they can sum to more than `total`.

    `total_edges` is the whole table, for scale — "12 dangling" reads
    very differently against 40 edges than against 40,000.

    `by_type` is the dangling count per edge type, largest first.
    """

    total: int
    total_edges: int
    missing_src: int
    missing_dst: int
    by_type: tuple[tuple[str, int], ...] = ()


def diagnose_dangling_edges(store_path: str) -> DanglingEdgeStats | None:
    """Return dangling-edge counts for `store_path`, or None (#1375).

    Read-only, via a plain sqlite connection rather than a `MemoryStore`
    — opening a store *runs* its pending one-shot migrations, which a
    diagnostic must not do.

    Report-only by design. The structural fix is a foreign key on
    `edges`, which needs a table rebuild; an `edges` migration is what
    bricked stores in #1161, so this leaf counts and does not repair.

    Fail-soft: a missing file, a store with no `edges` table, or any
    sqlite error yields None and the section is not rendered.
    """
    if store_path == ":memory:" or not Path(store_path).exists():
        return None
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
        # One LEFT JOIN pair drives both queries: `beliefs.id` is the
        # primary key, so each probe is an index lookup rather than a
        # scan of the belief table.
        dangling_from = (
            "FROM edges e "
            "LEFT JOIN beliefs bs ON bs.id = e.src "
            "LEFT JOIN beliefs bd ON bd.id = e.dst "
            "WHERE bs.id IS NULL OR bd.id IS NULL"
        )
        row = conn.execute(
            "SELECT COUNT(*), SUM(bs.id IS NULL), SUM(bd.id IS NULL) "
            + dangling_from
        ).fetchone()
        total_row = conn.execute("SELECT COUNT(*) FROM edges").fetchone()
        by_type = conn.execute(
            "SELECT e.type, COUNT(*) AS n "
            + dangling_from
            + " GROUP BY e.type ORDER BY n DESC, e.type ASC"
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        if conn is not None:
            conn.close()
    if row is None or total_row is None:
        return None
    return DanglingEdgeStats(
        total=int(row[0] or 0),
        total_edges=int(total_row[0] or 0),
        missing_src=int(row[1] or 0),
        missing_dst=int(row[2] or 0),
        by_type=tuple((str(t), int(n)) for t, n in by_type),
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
    from aelfrice.hook_audit import (  # noqa: PLC0415
        read_user_prompt_submit_telemetry,
    )

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
    # #1375: edges whose src or dst names a belief that does not exist.
    # The `edges` table carries no foreign key and `insert_edge` does not
    # check endpoint existence, so nothing else reports these. None when
    # no store path was supplied or the store could not be read.
    dangling_edges: DanglingEdgeStats | None = None
    # Runtime deps declared in pyproject.toml that are not importable
    # in the current environment (issue #236: stale uv tool env).
    missing_runtime_deps: list[str] = field(
        default_factory=lambda: cast(list[str], [])
    )
    # Basenames of default-on manifest hooks absent from every scanned
    # settings.json. Installs that predate a default flip end up here
    # (#557): they have partial wiring and need a re-run. Since #1161 the
    # covered set is read from the bundled manifest rather than a
    # hand-maintained tuple, so it tracks the installer automatically.
    missing_auto_capture_hooks: list[str] = field(
        default_factory=lambda: cast(list[str], [])
    )
    # `aelf-*` hook entries installed more than once for the same
    # (event, matcher, basenames) — every one of them fires per event,
    # so a duplicate is pure doubled latency (#1161). Repaired by
    # `aelf doctor --prune`.
    duplicate_hook_entries: list[DuplicateHookEntry] = field(
        default_factory=lambda: cast(list["DuplicateHookEntry"], [])
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
    # #1161: one-shot store migrations that raised on a recent open,
    # mapping method name -> exception repr. Since #1161 such a failure
    # no longer prevents the store from opening — the pass is recorded
    # and skipped — so this report is the operator's only signal that
    # the store is running with an incomplete migration. Empty on a
    # healthy store. Populated only when `diagnose` is given a
    # `store_path` it can open.
    failed_store_migrations: dict[str, str] = field(
        default_factory=lambda: cast(dict[str, str], {})
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
    # #1359: whether the UserPromptSubmit hook will write the
    # <aelfrice-memory> block, resolved from AELFRICE_MEMORY_BLOCK and
    # `[memory_block] enabled`. "Is this thing on?" is the question the
    # block's own hint line points here to answer, so the row is
    # unconditional on both `format_report` paths rather than rendered
    # only when the state is interesting. None is the one silent case
    # and means the probe itself failed — `aelfrice.hook` would not
    # import — which is a broken install doctor reports elsewhere.
    memory_block_enabled: bool | None = None
    # #1622: typed `/aelf:lock` requests that failed and are still not
    # locked. Rendered on both `format_report` paths whatever its value.
    # None means no store path was supplied, which renders as unknown;
    # the report's own `known` flag covers a disabled audit and an
    # unreadable store. Informational: it never changes the exit code.
    lock_gaps: "LockGapReport | None" = None
    # #1652: `$HOME/.aelfrice.toml` when it exists and the project has no
    # config of its own, so its settings silently do not apply (#1582).
    # A warning only: it never changes the exit code.
    ignored_home_config: Path | None = None
    # #1657: the claude-memory mirror hook is installed and the project has
    # memory to mirror, but nothing has turned the mirror on (no env, no
    # TOML, no consent sentinel), so it never writes. A warning only.
    mirror_consent_missing: bool = False

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
    store_path: str | None = None,
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
    `store_path` enables the incomplete-migration block (#1161) — pass
    the resolved DB path so doctor can read the store's failed one-shot
    migrations. Omitted (or unopenable) leaves that block quiet.
    """
    # Both home-derived paths are read through `_setup` rather than
    # copied in at import: doctor used to hold a by-value alias of
    # USER_SETTINGS_PATH and its OWN second constant named
    # SLASH_COMMANDS_DIR_DEFAULT, so a test patching setup's globals
    # reached neither (#1320).
    user_path = (
        user_settings if user_settings is not None
        else _setup.USER_SETTINGS_PATH
    )
    project_path = (
        project_root if project_root is not None else Path.cwd()
    ) / PROJECT_SETTINGS_RELPATH
    report = DoctorReport()
    if user_path.exists():
        report.scopes_scanned.append(("user", user_path))
        report.findings.extend(_scan_settings(user_path))
        report.duplicate_hook_entries.extend(
            find_duplicate_hook_entries(user_path)
        )
    if project_path.exists():
        report.scopes_scanned.append(("project", project_path))
        report.findings.extend(_scan_settings(project_path))
        report.duplicate_hook_entries.extend(
            find_duplicate_hook_entries(project_path)
        )
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
    # #1161: read incomplete one-shot migrations off the active store.
    # Fail-soft on every error: doctor's job here is to report, and a
    # store that cannot be opened at all is already covered by the
    # legacy-schema and graph-health blocks.
    if store_path is not None:
        report.failed_store_migrations = _read_failed_store_migrations(
            store_path
        )
        # #1375: dangling-edge count, same read-only handle policy.
        report.dangling_edges = diagnose_dangling_edges(store_path)
        # #1622: unapplied typed locks, read-only, same policy. Guarded
        # here as well: the detector reports its known failures as
        # unknown, and anything it did not anticipate must not end the
        # doctor run either.
        try:
            report.lock_gaps = diagnose_lock_gaps(store_path, project_root)
        except Exception as exc:  # noqa: BLE001 - fail-soft section
            from aelfrice.lock_gaps import (  # noqa: PLC0415
                LockGapReport as _LockGapReport,
            )

            report.lock_gaps = _LockGapReport(
                known=False,
                unknown_reason=f"the check failed: {type(exc).__name__}",
            )
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
            else _setup.SLASH_COMMANDS_DIR_DEFAULT
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
    # #1652: a per-user config the bounded walk no longer reads.
    try:
        from aelfrice.config_discovery import ignored_home_config  # noqa: PLC0415

        report.ignored_home_config = ignored_home_config(
            project_root if project_root is not None else Path.cwd(),
        )
    except Exception:  # noqa: BLE001 - fail-soft section
        report.ignored_home_config = None
    # #1657: an installed mirror hook that consent never switched on. The
    # sentinel sits beside the store doctor opens (`db_path()`, from the cwd),
    # like every other store check here; `project_root` picks the memory
    # directory and the TOML.
    try:
        from aelfrice.claude_memory import (  # noqa: PLC0415
            derive_memory_dir,
            mirror_consent_missing,
        )
        from aelfrice.setup import CLAUDE_MEMORY_MIRROR_SCRIPT_NAME  # noqa: PLC0415

        root = project_root if project_root is not None else Path.cwd()
        hook_installed = any(
            CLAUDE_MEMORY_MIRROR_SCRIPT_NAME in f.command for f in report.findings
        )
        report.mirror_consent_missing = (
            hook_installed
            and derive_memory_dir(root).is_dir()
            and mirror_consent_missing(start=root)
        )
    except Exception:  # noqa: BLE001 - fail-soft section
        report.mirror_consent_missing = False
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
    # #1359: memory-block injection state. Resolved from `project_root`
    # so a project-local `.aelfrice.toml` is honoured, and unconditional
    # (unlike the HRR block) because there is no probe to gate on and the
    # off state is exactly what needs reporting.
    report.memory_block_enabled = _diagnose_memory_block(project_root)
    return report


def _diagnose_memory_block(project_root: Path | None) -> bool | None:
    """Resolve the #1359 memory-block switch, or None if unresolvable.

    Read from `aelfrice.hook_audit`, not `aelfrice.hook`, which defines
    nothing of its own here. Importing `aelfrice.hook` from doctor closed
    an import cycle, `hook -> cli -> doctor -> hook` (#1631); check with
    `scripts/import_cycles.py --assert-acyclic aelfrice.hook`.

    Scope caveat: the env half resolves from *doctor's* process. A user
    who sets `AELFRICE_MEMORY_BLOCK` only in settings.json's `env` block
    gives the hook a value this shell never sees, so doctor reports the
    TOML answer. The TOML half is exact — it is read from
    `project_root`, the same way the hook reads it.
    """
    try:
        from aelfrice.hook_audit import memory_block_enabled  # noqa: PLC0415

        return memory_block_enabled(start=project_root)
    except Exception:
        return None


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
            encoding="utf-8", errors="replace",
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

    Checks every `.md` file directly under `slash_dir` against `known`;
    there is no special-casing for any filename prefix (no bundled
    slash-command file uses an `aelf-*` naming pattern today). Returns
    sorted basenames so the CLI report is stable.
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
    # #1412: NOT `shlex.split`. Its POSIX mode treats a backslash as an
    # escape, so `C:\Scripts\aelf-hook.exe` tokenises to
    # `C:Scriptsaelf-hook.exe` -- a path that does not exist. The hook is
    # then classified broken, and `prune_broken_aelf_hooks` (which `aelf
    # setup` runs unconditionally) DELETES a working Windows install on the
    # next run. This is the destructive half of the issue.
    try:
        tokens = launcher.command_tokens(stripped)
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
    # #1412: a Windows absolute path contains no forward slash, so it fell
    # through to the bare-name branch below and was reported as "not on
    # $PATH" however correctly it was installed.
    if "/" in program or "\\" in program:
        # #1482: `_resolve_script` writes the resolved path unquoted, so a
        # space anywhere in it splits the command and `program` is a
        # fragment — `/home/first last/.venv/bin/aelf-hook` checks as
        # `/home/first`, which does not exist. That reported a healthy
        # install broken; harmless while the prune predicate could not
        # recognise a spaced path as `aelf-*` at all, and a deletion the
        # moment it could. Check the reading that names a real file, and
        # keep reporting the raw token when none does.
        resolved = launcher.existing_program_path(stripped)
        target = program if resolved is None else str(resolved)
        finding = _check_path(settings_path, location, command, target)
        return _with_silent_failure(finding) if silent else finding
    # Bare name -- $PATH lookup. Explicit `path=` (see launcher) keeps the
    # win32 current-directory search out of a diagnostic.
    resolved = launcher.which_on_path(program)
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

    `duplicates_per_event` / `total_duplicates_removed` count entries
    dropped by the #1161 duplicate collapse rather than the broken-path
    pass. They are reported separately because the two repairs answer
    different questions: a pruned entry pointed at a vanished venv, a
    collapsed one was a redundant copy that resolved perfectly well and
    was silently doubling the hook's per-event cost.
    """

    settings_path: Path
    removed_per_event: dict[str, int]
    total_removed: int
    duplicates_per_event: dict[str, int] = field(
        default_factory=lambda: cast(dict[str, int], {})
    )
    total_duplicates_removed: int = 0


def _entry_duplicate_key(entry: object) -> tuple[str | None, str] | None:
    """Dedupe key for an `aelf-*` hook entry, or None if not ours.

    The key is `(matcher, "\\x00"-joined sorted aelf-* basenames)`. Two
    entries under the same event with the same key are the same logical
    hook installed twice, even if their absolute paths differ (a venv
    move changes the path, not the basename).

    Returns None when the entry contains no `aelf-*` inner command, which
    is what keeps this from ever touching a user's own hooks — the
    duplicate collapse must be blind to `conversation-logger.sh` and
    friends even if the user has genuinely listed one twice. Non-aelf
    inner commands are excluded from the key rather than making the whole
    entry ineligible, so a combined entry is keyed on its aelf content
    alone but still cannot collide with a differently-composed one.
    """
    if not isinstance(entry, dict):
        return None
    entry_dict = cast(dict[str, object], entry)
    inner = entry_dict.get(_INNER_HOOKS_KEY)
    if not isinstance(inner, list):
        return None
    basenames: list[str] = []
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
        # #1412: same key derivation as setup/host_codex ownership. A
        # Windows launcher used to key on the entire command string, so two
        # installs of the same hook never collided and the duplicate
        # collapse silently did nothing. #1482: a POSIX install path
        # containing a space keyed on the fragment before the space, which
        # is not `aelf-*`, so the same collapse did nothing there either —
        # and worse, two *different* hooks in one event shared that
        # fragment. The candidate that carries our prefix is the key; a
        # command with none is still excluded, so a foreign entry cannot
        # be grouped with ours.
        base = next(
            (
                key
                for key in launcher.command_program_keys(stripped)
                if key.startswith(_AELF_HOOK_BASENAME_PREFIX)
            ),
            None,
        )
        if base is not None:
            basenames.append(base)
    if not basenames:
        return None
    matcher = entry_dict.get("matcher")
    matcher_key = matcher if isinstance(matcher, str) else None
    return (matcher_key, "\x00".join(sorted(basenames)))


@dataclass(frozen=True)
class DuplicateHookEntry:
    """One `(event, matcher, basename)` installed more than once."""

    settings_path: Path
    event: str
    matcher: str | None
    basenames: str
    count: int

    def describe(self) -> str:
        shown = self.basenames.replace("\x00", "+")
        where = f"{self.event}" if self.matcher is None else (
            f"{self.event}[matcher={self.matcher}]"
        )
        return f"{where} {shown} ×{self.count}"


def find_duplicate_hook_entries(
    settings_path: Path,
) -> list[DuplicateHookEntry]:
    """Report `aelf-*` hook entries installed more than once per event.

    #1161: nothing detected this. `_install_or_replace_entry` returned on
    the first `(matcher, basename)` match — so once a settings.json held
    two entries for one logical hook, `aelf setup` reported "already
    installed" forever and `--prune` (which only removes entries whose
    program path is *broken*) had no reason to look. The duplicated
    entries were byte-identical and resolved fine, so every check passed
    while every aelfrice hook ran twice per event. Confirmed in the field
    on the maintainer's machine across all ten default-on hooks.

    Read-only. `prune_broken_aelf_hooks` performs the repair.
    """
    if not settings_path.exists():
        return []
    try:
        data = _load_settings_json(settings_path)
    except (ValueError, OSError):
        return []
    hooks_obj = data.get(_HOOKS_KEY)
    if not isinstance(hooks_obj, dict):
        return []
    hooks_dict = cast(dict[str, object], hooks_obj)
    dupes: list[DuplicateHookEntry] = []
    for event_name, event_list in hooks_dict.items():
        if not isinstance(event_list, list):
            continue
        counts: dict[tuple[str | None, str], int] = {}
        for entry in cast(list[object], event_list):
            key = _entry_duplicate_key(entry)
            if key is None:
                continue
            counts[key] = counts.get(key, 0) + 1
        for (matcher, basenames), count in counts.items():
            if count > 1:
                dupes.append(DuplicateHookEntry(
                    settings_path=settings_path,
                    event=event_name,
                    matcher=matcher,
                    basenames=basenames,
                    count=count,
                ))
    return dupes


def prune_broken_aelf_hooks(
    settings_path: Path, *, dry_run: bool = False,
) -> HookPruneResult:
    """Drop hook entries whose `aelf-*` program no longer resolves.

    Walks every `hooks.<event>[i].hooks[j]` command in `settings_path`.
    An entry is removed when ALL of:

    * its inner command's program has a basename starting with `aelf-`
      (so `aelf-hook`, `aelf-stop-hook`, etc. are in scope; bare shell
      scripts, `bash`, custom integrations are not) — under any reading
      of a path an unquoted space split, since #1482;
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
        # #1161: transaction-aware, so a prune running inside
        # `aelf setup`'s settings transaction mutates the same buffered
        # document instead of writing around it.
        data = read_settings(settings_path)
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
    duplicates_per_event: dict[str, int] = {}
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
        # #1161: collapse `aelf-*` entries installed more than once for
        # the same (matcher, basenames). Runs after the broken-path pass
        # so a broken duplicate is removed as broken and the surviving
        # copy is the one kept. First occurrence wins, preserving the
        # user's relative hook ordering within the event.
        seen: set[tuple[str | None, str]] = set()
        deduped: list[object] = []
        n_duplicates = 0
        for entry in kept:
            key = _entry_duplicate_key(entry)
            if key is not None:
                if key in seen:
                    n_duplicates += 1
                    continue
                seen.add(key)
            deduped.append(entry)
        if n_removed:
            removed_per_event[event_name] = n_removed
        if n_duplicates:
            duplicates_per_event[event_name] = n_duplicates
        if n_removed or n_duplicates:
            event_list[:] = deduped
    total = sum(removed_per_event.values())
    total_duplicates = sum(duplicates_per_event.values())
    if (total or total_duplicates) and not dry_run:
        write_settings(settings_path, data)
    return HookPruneResult(
        settings_path=settings_path,
        removed_per_event=removed_per_event,
        total_removed=total,
        duplicates_per_event=duplicates_per_event,
        total_duplicates_removed=total_duplicates,
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
        # #1412: the predicate that decides whether prune may delete this
        # entry. Under the old derivation a Windows launcher did not read as
        # `aelf-*`, so the entry was skipped -- benign. The damage came from
        # `_inspect_command` above, which then judged it broken. #1482: a
        # spaced POSIX path was skipped for the same reason, so a genuinely
        # broken one could never be pruned. Deletion still needs both
        # halves: an `aelf-*` candidate *and* a broken verdict from
        # `_inspect_command`, whose probe is the platform-neutral
        # `program_exists`.
        if not any(
            key.startswith(_AELF_HOOK_BASENAME_PREFIX)
            for key in launcher.command_program_keys(stripped)
        ):
            continue
        finding = _inspect_command(settings_path, "<prune>", cmd)
        if finding.status == "broken":
            return True
    return False


_HOOK_TYPE_COMMAND: Final[str] = "command"


def _atomic_rewrite_settings(path: Path, data: dict[str, object]) -> None:
    """Atomically replace `path` with the new JSON. Mirrors setup._atomic_write.

    The format matches setup's writer byte-for-byte (indent=2,
    ensure_ascii=False, trailing newline) so a setup→prune→setup round
    trip stays diff-stable.

    #1161: no longer used by the hook prune, which now goes through
    `setup.write_settings` so it can join an open settings transaction
    instead of writing around it. Retained as the unlocked writer for
    any future doctor repair that is genuinely independent of setup's
    transaction; a caller that mutates hooks should prefer
    `setup.write_settings`.
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
        # #1161: so is the store's migration state, and an install with
        # no settings.json is exactly when the store may be the problem.
        _format_failed_migrations_section(report, lines)
        # #1375: store-derived, independent of settings.json.
        _format_dangling_edges_section(report, lines)
        # #1359: config-derived, independent of settings.json.
        _format_memory_block_section(report, lines)
        # #1622: store- and audit-derived, independent of settings.json.
        _format_lock_gaps_section(report, lines)
        # #1652: config-derived, independent of settings.json.
        _format_ignored_home_config_section(report, lines)
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
    _format_duplicate_hook_entries_section(report, lines)
    _format_legacy_schema_section(report, lines)
    _format_failed_migrations_section(report, lines)
    _format_dangling_edges_section(report, lines)
    _format_hrr_section(report, lines)
    _format_memory_block_section(report, lines)
    _format_lock_gaps_section(report, lines)
    _format_ignored_home_config_section(report, lines)
    _format_mirror_consent_section(report, lines)
    return "\n".join(lines)


def _format_mirror_consent_section(report: DoctorReport, lines: list[str]) -> None:
    """Warn when the mirror hook is installed but never switched on (#1657)."""
    if not report.mirror_consent_missing:
        return
    lines.append("")
    lines.append("warning: the claude-memory mirror hook is installed but the mirror is off")
    lines.append(
        "  consent was never recorded for this project, so memory you write is "
        "not mirrored into aelfrice."
    )
    lines.append(
        "  fix: run `aelf reconcile-claude-memory` from this project, or set "
        "[memory] mirror_claude_memory in .aelfrice.toml to decide either way."
    )


def _format_ignored_home_config_section(
    report: DoctorReport, lines: list[str],
) -> None:
    """Append a warning when `$HOME/.aelfrice.toml` is ignored (#1652).

    Rendered only when there is something to say: most installs have no
    per-user file.
    """
    path = report.ignored_home_config
    if path is None:
        return
    lines.append("")
    lines.append(f"warning: {path} is ignored")
    lines.append(
        "  aelfrice reads only a project .aelfrice.toml (#1582), and this "
        "project has none."
    )
    lines.append(
        "  fix: copy it to the project's root to keep its settings."
    )


# Last-resort basenames if the bundled manifest cannot be read. Kept
# deliberately short — this is a floor, not a mirror. The live set comes
# from `_default_on_hook_basenames()` below.
_AUTO_CAPTURE_HOOK_BASENAMES_FALLBACK: Final[tuple[str, ...]] = (
    "aelf-hook",
    "aelf-transcript-logger",
    "aelf-commit-ingest",
    "aelf-session-start-hook",
    "aelf-stop-hook",
)


def _default_on_hook_basenames() -> tuple[str, ...]:
    """Basenames of every default-on manifest hook the user has not opted out of.

    #1161: this used to be a hand-maintained 4-tuple covering the v2.1
    auto-capture hooks, while the manifest had grown to ten default-on
    entries. The six it omitted included `aelf-hook` itself — the
    UserPromptSubmit retrieval hook, which is the entire product — so a
    settings.json that had lost it reported clean. The drift test was a
    tautology: it compared the tuple against the same `setup.*_SCRIPT_NAME`
    constants the tuple was copied from, never against the manifest.

    Reading `load_manifest()` makes the installer the single authority:
    a hook added to the manifest is covered by doctor the same release it
    ships, with no second list to remember. Opted-out hooks are excluded
    so a deliberate `--no-*` choice does not read as breakage.

    Returns basenames in manifest order, deduplicated: `search_tool` and
    `search_tool_bash` are distinct manifest rows that install the same
    program under different matchers, so basename presence cannot
    distinguish them. That granularity limit is why this reports
    "installed at all" rather than "installed for every matcher";
    `_find_duplicate_hook_entries` is the matcher-aware half.

    Fail-soft to `_AUTO_CAPTURE_HOOK_BASENAMES_FALLBACK` if the manifest
    is unreadable — `aelf doctor` must still run on a broken install.
    """
    try:
        from aelfrice.auto_install import (  # noqa: PLC0415
            load_manifest,
            read_opt_outs,
        )

        manifest = load_manifest()
        opted_out = read_opt_outs()
    except Exception:  # noqa: BLE001 — doctor must never fail to report
        return _AUTO_CAPTURE_HOOK_BASENAMES_FALLBACK
    basenames: list[str] = []
    for hook in manifest.hooks:
        if not hook.default_on or hook.name in opted_out:
            continue
        if hook.basename not in basenames:
            basenames.append(hook.basename)
    return tuple(basenames)


def _check_auto_capture_hooks(
    findings: list[CommandFinding],
) -> list[str]:
    """Return basenames of default-on manifest hooks absent from all scopes.

    A hook is "present" if any finding's command string contains the
    basename. Substring match is intentional: the command may be a bare
    basename (PATH-resolved install) or an absolute path (project venv);
    both should count as installed (#557).
    """
    missing: list[str] = []
    for basename in _default_on_hook_basenames():
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
        "default-on hooks not installed (#529, #1161). missing: "
        + ", ".join(report.missing_auto_capture_hooks)
    )
    lines.append(
        "fix: re-run 'aelf setup' to wire every default-on hook. to opt "
        "out per-hook, use the matching `aelf setup --no-*` flag (an "
        "opted-out hook is not reported here)."
    )


def _format_duplicate_hook_entries_section(
    report: DoctorReport, lines: list[str],
) -> None:
    """Append the #1161 duplicate-hook block to `lines`.

    Quiet when nothing is duplicated. Worth a loud line when it is: a
    duplicate resolves fine and breaks nothing, so the only symptom is
    that every aelfrice hook runs N times per event and the user's
    prompts get slower with no explanation.
    """
    if not report.duplicate_hook_entries:
        return
    lines.append("")
    total_extra = sum(d.count - 1 for d in report.duplicate_hook_entries)
    lines.append(
        f"duplicate aelf-* hook entries: {total_extra} redundant "
        f"entr{'y' if total_extra == 1 else 'ies'} (#1161). each copy "
        f"fires per event, so this is pure added latency:"
    )
    for dupe in sorted(
        report.duplicate_hook_entries,
        key=lambda d: (str(d.settings_path), d.event, d.basenames),
    ):
        lines.append(f"  {dupe.describe()}  [{dupe.settings_path}]")
    lines.append("fix: run 'aelf doctor --prune' to collapse them.")


def _read_failed_store_migrations(store_path: str) -> dict[str, str]:
    """Return `{migration_name: error_repr}` for `store_path`, or `{}`.

    #1161. Reads the `migration_failed:*` rows written by
    `MemoryStore._run_guarded_migration` with a plain read-only sqlite
    connection rather than by constructing a `MemoryStore`. Opening a
    store *runs* its pending one-shot migrations, which would make a
    read-only diagnostic mutate the DB and pay the migration cost, and
    would race any concurrent writer for the write lock.

    Fail-soft: a missing file, a missing `schema_meta` table (pre-v1.3
    store), or any sqlite error yields `{}`. Reporting nothing is the
    correct degradation for a diagnostic.
    """
    if store_path == ":memory:" or not Path(store_path).exists():
        return {}
    from aelfrice.store import SCHEMA_META_MIGRATION_FAILED_PREFIX

    prefix = SCHEMA_META_MIGRATION_FAILED_PREFIX
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"file:{store_path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT key, value FROM schema_meta WHERE key LIKE ? "
            "ORDER BY key ASC",
            (f"{prefix}%",),
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        if conn is not None:
            conn.close()
    return {str(k)[len(prefix):]: str(v) for k, v in rows}


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


def _format_failed_migrations_section(
    report: DoctorReport, lines: list[str],
) -> None:
    """Append the incomplete-store-migration block to `lines` (#1161).

    Quiet on a healthy store. When a one-shot migration raised, the
    store still opened (that is the point of the #1161 guard), so this
    block is the only place the operator learns the store is running in
    a degraded shape and what the underlying error was.
    """
    if not report.failed_store_migrations:
        return
    lines.append("")
    lines.append(
        "store migration(s) INCOMPLETE — the store opens and is usable, "
        "but one or more one-shot migrations could not finish:"
    )
    for name, err in sorted(report.failed_store_migrations.items()):
        lines.append(f"  {name}: {err}")
    lines.append(
        "fix: these retry automatically on every open, so upgrading "
        "(`aelf upgrade`) is the first thing to try. If the same "
        "migration keeps failing, report the error above — the store "
        "will keep working in the meantime."
    )


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


def _format_memory_block_section(
    report: DoctorReport, lines: list[str],
) -> None:
    """Append the memory-block injection state to `lines` (#1359).

    Quiet when the probe did not resolve. The disabled row names both
    spellings of the switch, because a user reaching doctor to ask why
    nothing is being injected needs the answer here, not in the docs.
    """
    if report.memory_block_enabled is None:
        return
    lines.append("")
    lines.append("Memory block")
    if report.memory_block_enabled:
        lines.append("  injection:            enabled")
    else:
        lines.append(
            "  injection:            disabled "
            "(AELFRICE_MEMORY_BLOCK=0 or [memory_block] enabled = false)"
        )


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


def _format_lock_gaps_section(report: DoctorReport, lines: list[str]) -> None:
    """Append the unapplied-typed-lock block to `lines` (#1622).

    Always rendered. Zero is printed as zero only when the detector could
    tell; otherwise the section says unknown and why.
    """
    st = report.lock_gaps
    lines.append("")
    lines.append("typed /aelf:lock requests that did not take effect:")
    if st is None:
        lines.append("  unknown: no store was checked")
        return
    if not st.known:
        lines.append(f"  unknown: {st.unknown_reason}")
        return
    since = f" since {st.first_ts}" if st.first_ts else ""
    if not st.gaps:
        lines.append(f"  none ({st.records_seen} recorded outcome(s){since})")
    else:
        lines.append(
            f"  {len(st.gaps)} still not locked "
            f"({st.records_seen} recorded outcome(s){since}):"
        )
        for g in st.gaps:
            tries = f", {g.attempts} attempts" if g.attempts > 1 else ""
            lines.append(
                f"  - {g.ts} ({g.reason}{tries}): {g.display_statement}"
            )
            fix = g.fix_command
            if fix is not None:
                lines.append(f"      fix: {fix}")
            elif g.truncated:
                # No runnable command: one built from the prefix would
                # lock text the user never typed (#1622).
                lines.append(
                    f"      statement shown is the first {len(g.statement)} "
                    f"of {g.arg_len} characters; to fix it, run "
                    f"`aelf lock` with the full statement"
                    + (", quoted for your shell" if _lock_gaps_on_windows() else "")
                )
            elif not g.valid_text:
                lines.append(
                    "      statement contains an unpaired surrogate and "
                    "cannot be locked; retype it"
                )
            else:
                # No runnable command on Windows: no single quoting is safe
                # in both cmd.exe and PowerShell (#1622).
                lines.append(
                    "      to fix it, run `aelf lock` with the statement "
                    "above, quoted for your shell"
                )
    lines.append(
        "  only requests typed since this check shipped are recorded; "
        "earlier failures are not listed."
    )


def _format_dangling_edges_section(
    report: DoctorReport, lines: list[str],
) -> None:
    """Append the dangling-edge block to `lines` (#1375).

    Rendered whenever the store could be read, including at zero — a
    check that is silent when clean is indistinguishable from a check
    that never ran, and "0 dangling" is the reading an operator wants
    confirmed before trusting a graph walk.
    """
    st = report.dangling_edges
    if st is None:
        return
    lines.append("")
    lines.append("dangling edges (endpoint missing from `beliefs`):")
    if st.total == 0:
        lines.append(f"  none of {st.total_edges} edge(s)")
        return
    lines.append(
        f"  {st.total} of {st.total_edges} edge(s) — "
        f"{st.missing_src} missing src, {st.missing_dst} missing dst"
    )
    for edge_type, n in st.by_type[:DANGLING_EDGE_TYPES_SHOWN]:
        lines.append(f"    {edge_type}: {n}")
    remainder = len(st.by_type) - DANGLING_EDGE_TYPES_SHOWN
    if remainder > 0:
        lines.append(f"    ... and {remainder} more type(s)")
    lines.append(
        "  cause: `edges` has no foreign key and `insert_edge` does not "
        "check that its endpoints exist, so a producer writing against a "
        "deleted or never-inserted belief id succeeds silently. These "
        "rows are not inert: BFS spends a `nodes_per_hop` slot on the "
        "neighbour before it tries to load it, so a dangling edge can "
        "displace a real one, and the edge-type rerank reads incoming "
        "edges by `dst` without checking that `src` exists, so a "
        "dangling POTENTIALLY_STALE row still demotes its target. They "
        "also inflate every edge count. Report-only: repairing them "
        "needs an `edges` table rebuild, which is out of scope here "
        "(#1161)."
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
        # and timestamps intact so prior work is not lost.
        updated = _belief_with_type(belief, cls.belief_type, ORIGIN_AGENT_INFERRED)
        store.update_belief(updated)
        report.classified += 1

    report.type_dist_after = store.count_beliefs_by_type()
    return report


def _belief_with_type(
    b: "object", new_type: str, new_origin: str
) -> "object":
    """Return a modified copy of `b` (a Belief) with `type` and `origin`
    replaced.

    Does NOT use dataclasses.replace: only id, content, content_hash,
    alpha, beta, type, lock_level, locked_at, created_at,
    last_retrieved_at, session_id, and origin are carried over.
    corroboration_count, hibernation_score, activation_condition,
    retention_class, valid_to, scope, project_context,
    last_confirmed_at, and lock_tier are silently reset to their
    dataclass defaults on every call. `store.update_belief()`'s
    subsequent full-row UPDATE then clobbers the stored values of all
    of those except corroboration_count (which isn't part of that
    UPDATE). Use `dataclasses.replace(belief, type=new_type,
    origin=new_origin)` instead if full-fidelity preservation is
    required. The import is local to keep the doctor module importable
    without triggering the full models chain at top level.
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
# gc-filesystem-corroboration pass (issue #1669)
# ---------------------------------------------------------------------------


@dataclass
class FilesystemCorroborationReport:
    """Summary of one `gc_filesystem_corroboration` pass.

    `rows_found` is the number of corroboration rows from non-asserting
    sources, spread over `beliefs_affected` beliefs. `leaving_core` lists
    the active, unlocked beliefs that meet the `aelf core` rule now and
    don't once those rows are gone. `entering_core` lists the ones that
    don't now and do once the rows are gone. Both are computed by running
    the delete and reading core back, so a dry run reports the same ids an
    apply would. `deleted` is zero on a dry run.

    Core here includes the #1638 admission gate, which is not monotone in
    corroboration: a B-labeled belief is admitted only with fewer than two
    episode-qualified corroborations, so deleting rows can move it into
    core. Removing a sighting can also merge two short gaps into one long
    enough to start a new episode, so the episode rule alone can admit a
    belief too.
    """

    rows_found: int = 0
    beliefs_affected: int = 0
    leaving_core: list[str] = field(default_factory=list[str])
    entering_core: list[str] = field(default_factory=list[str])
    deleted: int = 0
    dry_run: bool = True


class _DryRunRollback(Exception):
    """Raised inside the transaction to undo a dry run's delete."""


def _core_members(
    store: "MemoryStore",
    belief_ids: list[str],
    qualifies: Callable[["Belief", int], bool],
) -> set[str]:
    """The subset of `belief_ids` in core: admitted by `qualifies` and
    then by the #1638 admission gate (`MemoryStore.gate_core_candidates`),
    as `aelf core` selects them.

    Locked beliefs are left out: they're in core through the lock arm,
    which corroboration rows never affect.
    """
    from aelfrice.models import LOCK_NONE  # noqa: PLC0415

    episodes = store.corroboration_episodes()
    qualifying: list["Belief"] = []
    for bid in belief_ids:
        b = store.get_belief(bid)
        if b is None or b.lock_level != LOCK_NONE:
            continue
        if qualifies(b, episodes.get(bid, 0)):
            qualifying.append(b)
    admitted, _ = store.gate_core_candidates(qualifying, episodes)
    return {b.id for b in admitted}


def gc_filesystem_corroboration(
    store: "MemoryStore",
    *,
    qualifies: Callable[["Belief", int], bool],
    dry_run: bool = True,
) -> FilesystemCorroborationReport:
    """Count and (with `dry_run=False`) delete corroboration rows from
    non-asserting sources. Issue #1669.

    Before #1615, every onboard or repository scan re-read the same
    files under a new scan session and wrote a corroboration row for
    each paragraph it found again. Three scans gave every doc belief a
    count of 2, and enough wall-clock spread to pass the episode rule.
    #1615 stopped the writes; this pass removes the rows already
    stored. Rows of every other source are untouched.

    `qualifies(belief, episodes)` is the unlocked `aelf core` rule. The
    caller passes it, because the rule lives in `cli`, which imports
    this module.

    The dry run deletes inside a transaction, reads core membership
    back, and rolls the transaction back, so its `leaving_core` is
    measured rather than estimated. Every read happens under the write
    lock, so a concurrent writer can't change core between the two
    reads. The pass must own that transaction: it raises `RuntimeError`
    when one is already open on `store`, where the rollback would either
    not happen or discard the caller's writes.
    """
    from aelfrice.models import CORROBORATION_SOURCES_NON_ASSERTING  # noqa: PLC0415

    if store.transaction_open:
        raise RuntimeError(
            "gc_filesystem_corroboration needs its own transaction; "
            "call it outside store.transaction() and with no pending writes"
        )
    sources = CORROBORATION_SOURCES_NON_ASSERTING
    report = FilesystemCorroborationReport(dry_run=dry_run)
    try:
        with store.transaction(immediate=True):
            per_belief = store.count_corroborations_by_source(sources)
            report.rows_found = sum(per_belief.values())
            report.beliefs_affected = len(per_belief)
            if report.rows_found == 0:
                return report
            affected = sorted(per_belief)
            before = _core_members(store, affected, qualifies)
            deleted = store.delete_corroborations_by_source(sources)
            after = _core_members(store, affected, qualifies)
            report.leaving_core = sorted(before - after)
            report.entering_core = sorted(after - before)
            if dry_run:
                raise _DryRunRollback
    except _DryRunRollback:
        return report
    report.deleted = deleted
    return report


def format_filesystem_corroboration_report(
    report: FilesystemCorroborationReport,
) -> str:
    """Human-readable rendering of `gc_filesystem_corroboration` output."""
    lines: list[str] = [
        f"filesystem corroboration rows: {report.rows_found} "
        f"on {report.beliefs_affected} belief(s)",
        f"beliefs leaving `aelf core`: {len(report.leaving_core)}",
    ]
    for bid in report.leaving_core[:15]:
        lines.append(f"  {bid}")
    if len(report.leaving_core) > 15:
        lines.append(f"  ... and {len(report.leaving_core) - 15} more")
    # #1638: printed only when non-empty, so a report with nothing entering
    # core reads as it did before the gate.
    if report.entering_core:
        lines.append(f"beliefs entering `aelf core`: {len(report.entering_core)}")
        for bid in report.entering_core[:15]:
            lines.append(f"  {bid}")
        if len(report.entering_core) > 15:
            lines.append(f"  ... and {len(report.entering_core) - 15} more")
    if report.dry_run:
        if report.rows_found == 0:
            lines.append("nothing to do.")
        else:
            lines.append("dry-run; re-run with --apply to delete these rows.")
    else:
        lines.append(f"deleted: {report.deleted}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# core admission gate backlog (issue #1638)
# ---------------------------------------------------------------------------


def core_gate_candidates(
    store: "MemoryStore",
    qualifies: Callable[["Belief", int], bool],
) -> tuple[list["Belief"], dict[str, str]]:
    """The beliefs the core admission gate judges, with their labels.

    Returns the active, unlocked beliefs that `qualifies` admits to core
    (today's non-lock rule, before the gate), in ascending id order, and
    their labels under the current `CLASSIFIER_VERSION`, keyed by content
    hash, read in one lookup. A candidate without an entry is unlabeled.
    Locked beliefs are left out: the gate doesn't apply to them.
    """
    from aelfrice.core_gate import CLASSIFIER_VERSION  # noqa: PLC0415
    from aelfrice.models import LOCK_NONE  # noqa: PLC0415

    episodes = store.corroboration_episodes()
    candidates: list["Belief"] = []
    for bid in store.list_belief_ids():
        b = store.get_belief(bid)
        if b is None or b.lock_level != LOCK_NONE:
            continue
        if qualifies(b, episodes.get(bid, 0)):
            candidates.append(b)
    labels = store.core_gate_labels_for(
        (b.content_hash for b in candidates), CLASSIFIER_VERSION,
    )
    return candidates, labels


@dataclass
class CoreGateCoverage:
    """How much of core's gated membership has a label (#1638).

    `candidates` counts the active, unlocked beliefs that meet today's
    non-lock core rule. `labeled` counts those with a label under
    `classifier_version`, per label; `unlabeled` counts the rest, which
    is the backlog `aelf doctor core-gate --emit` batches. Informational:
    it never changes doctor's exit code.
    """

    classifier_version: str
    candidates: int = 0
    labeled: dict[str, int] = field(
        default_factory=lambda: {"A": 0, "B": 0, "C": 0},
    )
    unlabeled: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "classifier_version": self.classifier_version,
            "candidates": self.candidates,
            "labeled": dict(self.labeled),
            "unlabeled": self.unlabeled,
        }


def core_gate_coverage(
    store: "MemoryStore",
    qualifies: Callable[["Belief", int], bool],
) -> CoreGateCoverage:
    """Count labeled and unlabeled core candidates (#1638). Read-only."""
    from aelfrice.core_gate import CLASSIFIER_VERSION  # noqa: PLC0415

    candidates, labels = core_gate_candidates(store, qualifies)
    report = CoreGateCoverage(classifier_version=CLASSIFIER_VERSION)
    report.candidates = len(candidates)
    for b in candidates:
        label = labels.get(b.content_hash)
        if label is None:
            report.unlabeled += 1
        else:
            report.labeled[label] = report.labeled.get(label, 0) + 1
    return report


def format_core_gate_coverage(report: CoreGateCoverage) -> list[str]:
    """The core-gate coverage block, as lines."""
    a, b, c = (report.labeled.get(k, 0) for k in ("A", "B", "C"))
    lines = [
        f"core admission gate ({report.classifier_version}):",
        f"  {report.candidates} unlocked core candidates: "
        f"{a + b + c} labeled (A {a}, B {b}, C {c}), "
        f"{report.unlabeled} unlabeled",
    ]
    if report.unlabeled:
        lines.append(
            "  unlabeled candidates follow today's rule; run "
            "`aelf doctor core-gate --emit` to batch them for the classifier."
        )
    return lines


@dataclass(frozen=True)
class CoreGateEmitBatch:
    """One batch `aelf doctor core-gate --emit` prints (#1638).

    `reused` is True for a batch an earlier emit created that isn't
    accepted yet, printed again rather than batched a second time.
    """

    batch_id: str
    prompt: str
    size: int
    created_at: str
    reused: bool


@dataclass
class CoreGateEmitReport:
    """The result of one `aelf doctor core-gate --emit` call (#1638).

    `backlog` counts the unlabeled core candidates. `batches` are the
    batches to print, the reused ones first. `left` counts the backlog
    beliefs in no printed batch, because `--limit` stopped the run.
    """

    classifier_version: str
    backlog: int = 0
    batches: list[CoreGateEmitBatch] = field(
        default_factory=list[CoreGateEmitBatch],
    )
    left: int = 0


_T = TypeVar("_T")


def balanced_chunks(items: list[_T], max_size: int) -> list[list[_T]]:
    """Split `items`, in order, into ceil(len / max_size) consecutive
    chunks whose sizes differ by at most one, the larger ones first.

    151 items at a maximum of 50 give 38, 38, 38, 37, not 50, 50, 50, 1.
    The self-check compares each batch's share of C labels with its
    siblings', and a remainder batch of one or a few snippets can only
    score a share near 0 or 1, so it would be flagged on healthy runs.
    """
    if not items:
        return []
    n = -(-len(items) // max_size)
    size, extra = divmod(len(items), n)
    chunks: list[list[_T]] = []
    start = 0
    for k in range(n):
        end = start + size + (1 if k < extra else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


def emit_core_gate_batches(
    store: "MemoryStore",
    qualifies: Callable[["Belief", int], bool],
    *,
    limit: int | None,
    created_at: str,
) -> CoreGateEmitReport:
    """Batch the core-gate backlog for the host's classifier (#1638).

    The backlog is the active, unlocked beliefs that meet today's non-lock
    core rule (`qualifies`) and have no label under the current
    `CLASSIFIER_VERSION`, in ascending id order.

    A second emit doesn't batch a belief twice. A doctor batch from an
    earlier emit that isn't accepted yet is printed again, unchanged, when
    every one of its items is still in the backlog under the same content
    hash; its beliefs go into no new batch. An open batch with any item
    that has since been labeled, retired, locked, or changed, or that no
    longer meets the rule, is set aside: it's left as it is, and its
    remaining backlog beliefs go into new batches.

    New batches split the rest of the backlog evenly (`balanced_chunks`):
    as few batches as `core_gate.MAX_BATCH` allows, whose sizes differ by
    at most one, in backlog order. They're created with origin `doctor`, no session, and the one
    `created_at` this call is given, which marks them as one emit run for
    the accept-time self-check. All of them are written in one
    transaction; nothing else is written.

    `limit` caps the number of batches printed, the reused ones first
    (oldest first) and then the new ones; a new batch past the cap isn't
    created. None means no cap.
    """
    from aelfrice.core_gate import (  # noqa: PLC0415
        CLASSIFIER_VERSION,
        MAX_BATCH,
        build_prompt,
    )
    from aelfrice.models import (  # noqa: PLC0415
        CORE_GATE_ORIGIN_DOCTOR,
        CoreGateBatchItem,
    )

    candidates, labels = core_gate_candidates(store, qualifies)
    backlog = [b for b in candidates if b.content_hash not in labels]
    by_hash = {b.content_hash: b for b in backlog}
    report = CoreGateEmitReport(
        classifier_version=CLASSIFIER_VERSION, backlog=len(backlog),
    )

    claimed: set[str] = set()
    reused: list[CoreGateEmitBatch] = []
    for batch in store.list_open_core_gate_batches(
        origin=CORE_GATE_ORIGIN_DOCTOR, classifier_version=CLASSIFIER_VERSION,
    ):
        items = sorted(batch.items, key=lambda i: i.index)
        if not all(
            item.content_hash in by_hash
            and item.content_hash not in claimed
            for item in items
        ):
            continue
        claimed.update(item.content_hash for item in items)
        reused.append(CoreGateEmitBatch(
            batch_id=batch.batch_id,
            prompt=build_prompt([
                (item.index, by_hash[item.content_hash].content)
                for item in items
            ]),
            size=len(items),
            created_at=batch.created_at,
            reused=True,
        ))

    fresh = [b for b in backlog if b.content_hash not in claimed]
    chunks = balanced_chunks(fresh, MAX_BATCH)
    if limit is not None:
        reused = reused[:limit]
        chunks = chunks[:max(0, limit - len(reused))]

    created: list[CoreGateEmitBatch] = []
    with store.transaction():
        for chunk in chunks:
            batch_id = store.create_core_gate_batch(
                [
                    CoreGateBatchItem(
                        index=i, belief_id=b.id, content_hash=b.content_hash,
                    )
                    for i, b in enumerate(chunk)
                ],
                classifier_version=CLASSIFIER_VERSION,
                origin=CORE_GATE_ORIGIN_DOCTOR,
                session_id=None,
                created_at=created_at,
            )
            created.append(CoreGateEmitBatch(
                batch_id=batch_id,
                prompt=build_prompt([(i, b.content) for i, b in enumerate(chunk)]),
                size=len(chunk),
                created_at=created_at,
                reused=False,
            ))
    report.batches = reused + created
    report.left = report.backlog - sum(b.size for b in report.batches)
    return report


class CoreGateRerunRefused(ValueError):
    """`rerun_core_gate_batch` refused a batch; nothing was written."""


@dataclass
class CoreGateRerunReport:
    """The result of `aelf doctor core-gate --rerun <batch-id>` (#1638).

    `labels_dropped` counts the label rows the batch still owned and that
    were deleted. `kept` counts the batch's beliefs re-batched: still
    active, unlocked, in core under today's rule, unchanged, and still
    labeled by this batch. `batch` is the batch to print, or None when no
    belief was kept. `reopened` is True when `batch` is the original batch
    itself, reopened, and False when it is a new batch.
    """

    rerun_of: str
    classifier_version: str
    labels_dropped: int = 0
    kept: int = 0
    batch: CoreGateEmitBatch | None = None
    reopened: bool = False


def rerun_core_gate_batch(
    store: "MemoryStore",
    qualifies: Callable[["Belief", int], bool],
    batch_id: str,
) -> CoreGateRerunReport:
    """Re-run one accepted doctor batch, for the spec's self-check step 2.

    In one `BEGIN IMMEDIATE` transaction: drop the labels the batch still
    owns (`core_gate_labels.batch_id`; a hash a later batch relabeled is
    left alone), and batch again the beliefs whose labels were dropped and
    that are still active, unlocked, in core under `qualifies`, and under
    the same content hash.

    The fresh batch joins the original emit run, so the self-check
    compares it with the same siblings: it carries the original
    `created_at`. A batch id is derived from the classifier version, the
    `created_at`, and the content hashes, so when every belief is kept the
    fresh batch's id is the original's. That case reopens the original
    row (clears `accepted_at`) instead of inserting a duplicate. When
    some beliefs drop out, the smaller set derives a new id, and a new
    batch at the original `created_at` is created. Either way the original
    batch then owns no labels, so the self-check doesn't count it.

    Raises `CoreGateRerunRefused`, writing nothing, for an unknown batch,
    a batch not accepted, a batch from another classifier version, a
    session-end batch, or a batch that owns no labels any more (already
    re-run, or every label replaced by a later batch).
    """
    from aelfrice.core_gate import CLASSIFIER_VERSION, build_prompt  # noqa: PLC0415
    from aelfrice.models import (  # noqa: PLC0415
        CORE_GATE_ORIGIN_DOCTOR,
        LOCK_NONE,
        CoreGateBatchItem,
    )

    with store.transaction(immediate=True):
        batch = store.get_core_gate_batch(batch_id)
        if batch is None:
            raise CoreGateRerunRefused(f"no core-gate batch {batch_id}")
        if batch.origin != CORE_GATE_ORIGIN_DOCTOR:
            raise CoreGateRerunRefused(
                f"batch {batch_id} is a {batch.origin} batch; only a batch "
                "from `aelf doctor core-gate --emit` can be re-run"
            )
        if batch.accepted_at is None:
            raise CoreGateRerunRefused(
                f"batch {batch_id} was never accepted; accept it, or print "
                "it again with `aelf doctor core-gate --emit`"
            )
        if batch.classifier_version != CLASSIFIER_VERSION:
            raise CoreGateRerunRefused(
                f"batch {batch_id} was emitted under "
                f"{batch.classifier_version}, not the current "
                f"{CLASSIFIER_VERSION}; its labels no longer apply"
            )
        owned = store.core_gate_label_hashes_of_batch(
            batch_id, CLASSIFIER_VERSION,
        )
        if not owned:
            raise CoreGateRerunRefused(
                f"batch {batch_id} owns no labels: it was already re-run, "
                "or later batches relabeled all of its beliefs"
            )
        episodes = store.corroboration_episodes()
        items = sorted(batch.items, key=lambda i: i.index)
        kept: list[tuple["CoreGateBatchItem", "Belief"]] = []
        for item in items:
            if item.content_hash not in owned:
                continue
            b = store.get_belief(item.belief_id)
            if (
                b is None
                or b.lock_level != LOCK_NONE
                or b.content_hash != item.content_hash
                or not qualifies(b, episodes.get(b.id, 0))
            ):
                continue
            kept.append((item, b))
        report = CoreGateRerunReport(
            rerun_of=batch_id, classifier_version=CLASSIFIER_VERSION,
        )
        report.labels_dropped = store.delete_core_gate_labels_of_batch(
            batch_id, CLASSIFIER_VERSION,
        )
        report.kept = len(kept)
        if not kept:
            return report
        if len(kept) == len(items):
            store.reopen_core_gate_batch(batch_id)
            report.reopened = True
            new_id = batch_id
            snippets = [(item.index, b.content) for item, b in kept]
        else:
            try:
                new_id = store.create_core_gate_batch(
                    [
                        CoreGateBatchItem(
                            index=i, belief_id=b.id,
                            content_hash=b.content_hash,
                        )
                        for i, (_, b) in enumerate(kept)
                    ],
                    classifier_version=CLASSIFIER_VERSION,
                    origin=CORE_GATE_ORIGIN_DOCTOR,
                    session_id=None,
                    created_at=batch.created_at,
                )
            except ValueError as exc:
                raise CoreGateRerunRefused(str(exc)) from exc
            existing = store.get_core_gate_batch(new_id)
            if existing is not None and existing.accepted_at is not None:
                # An identical batch already exists and was accepted: the
                # store returned its id instead of a new row.
                raise CoreGateRerunRefused(
                    f"the re-run batch would be {new_id}, which is already "
                    "accepted"
                )
            snippets = [(i, b.content) for i, (_, b) in enumerate(kept)]
        report.batch = CoreGateEmitBatch(
            batch_id=new_id,
            prompt=build_prompt(snippets),
            size=len(kept),
            created_at=batch.created_at,
            reused=False,
        )
    return report


# ---------------------------------------------------------------------------
# repair-utc-created-at pass (issue #1660)
# ---------------------------------------------------------------------------


@dataclass
class UtcCreatedAtReport:
    """Summary of one `repair_utc_created_at` pass.

    `rows_found` counts beliefs whose `created_at` carries a non-zero UTC
    offset, spread over `sessions_affected` sessions. `log_rows_found`
    counts the `ingest_log.ts` values rewritten with them. The spine counts
    are the TEMPORAL_NEXT edges the re-chain removes and writes. Like the
    filesystem-corroboration pass, the dry run makes the change inside a
    transaction and rolls it back, so its counts are measured. `samples`
    holds up to five `(belief_id, before, after)` rows.
    """

    rows_found: int = 0
    log_rows_found: int = 0
    sessions_affected: int = 0
    spine_edges_removed: int = 0
    spine_edges_written: int = 0
    rewritten: int = 0
    dry_run: bool = True
    samples: list[tuple[str, str, str]] = field(
        default_factory=list[tuple[str, str, str]],
    )


def utc_z_form(value: str) -> str | None:
    """`value` rewritten as UTC with a `Z` suffix, or None to leave it.

    None when it doesn't parse, is naive, or is already UTC (`Z` or
    `+00:00`, the same instant). Fractional seconds are kept, so a value
    without them gets the `YYYY-MM-DDTHH:MM:SSZ` form the scanner has
    written since #1611.
    """
    try:
        dt = datetime.fromisoformat(value)
        offset = dt.utcoffset()
        if offset is None or offset == timedelta(0):
            return None
        utc = dt.astimezone(timezone.utc)
    except (ValueError, OverflowError):
        # Unparseable, or an offset that moves it past year 1 or 9999.
        return None
    return utc.isoformat().replace("+00:00", "Z")


def repair_utc_created_at(
    store: "MemoryStore",
    *,
    dry_run: bool = True,
) -> UtcCreatedAtReport:
    """Rewrite `created_at` values with a non-zero UTC offset as UTC `Z`,
    then re-chain the spine of each session they're in. Issue #1660.

    Before #1611 the scanner stored git author dates with the author's
    local offset. Next to `Z` rows, text order then differs from real
    order, and the spine, ordered by `(created_at, rowid)`, links some
    pairs backwards in time. The matching `ingest_log.ts` values are
    rewritten too, because derivation copies them into `created_at` and
    the log is the source of truth (#1283). A store with no such value
    is left untouched: the pass returns before it opens a transaction.

    The dry run holds the write lock while it runs, so hook writes wait
    for it (about 0.3 s at 12k rows).

    Must own its transaction, for the same reason as
    `gc_filesystem_corroboration`: it raises `RuntimeError` when one is
    already open on `store`.
    """
    from aelfrice.temporal_spine import rechain_sessions  # noqa: PLC0415

    if store.transaction_open:
        raise RuntimeError(
            "repair_utc_created_at needs its own transaction; "
            "call it outside store.transaction() and with no pending writes"
        )
    report = UtcCreatedAtReport(dry_run=dry_run)
    candidates = (
        store.created_at_with_numeric_offset()
        + store.ingest_log_ts_with_numeric_offset()
    )
    if not any(utc_z_form(v) is not None for _, v in candidates):
        return report
    try:
        with store.transaction(immediate=True):
            changes = [
                (bid, old, new)
                for bid, old in store.created_at_with_numeric_offset()
                if (new := utc_z_form(old)) is not None
            ]
            sessions: set[str] = set()
            for bid, _, new in changes:
                store.set_belief_created_at(bid, new)
                b = store.get_belief(bid, include_retired=True)
                if b is not None and b.session_id is not None:
                    sessions.add(b.session_id)
            log_changes = [
                (log_id, new)
                for log_id, old in store.ingest_log_ts_with_numeric_offset()
                if (new := utc_z_form(old)) is not None
            ]
            for log_id, new in log_changes:
                store.set_ingest_log_ts(log_id, new)
            report.rows_found = len(changes)
            report.log_rows_found = len(log_changes)
            report.sessions_affected = len(sessions)
            report.samples = changes[:5]
            removed, written = rechain_sessions(store, sessions)
            report.spine_edges_removed = removed
            report.spine_edges_written = written
            if dry_run:
                raise _DryRunRollback
    except _DryRunRollback:
        return report
    report.rewritten = report.rows_found
    return report


def format_utc_created_at_report(report: UtcCreatedAtReport) -> str:
    """Human-readable rendering of `repair_utc_created_at` output."""
    lines: list[str] = [
        f"created_at rows with a non-UTC offset: {report.rows_found} "
        f"in {report.sessions_affected} session(s)",
        f"ingest_log rows with a non-UTC offset: {report.log_rows_found}",
        f"spine edges re-chained: {report.spine_edges_removed} removed, "
        f"{report.spine_edges_written} written",
    ]
    for bid, before, after in report.samples:
        lines.append(f"  {bid}: {before} -> {after}")
    if report.dry_run:
        if report.rows_found == 0:
            lines.append("nothing to do.")
        else:
            lines.append("dry-run; re-run with --apply to rewrite these rows.")
    else:
        lines.append(f"rewritten: {report.rewritten}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# promote-retention pass (issue #290 phase-3)
# ---------------------------------------------------------------------------

# Promotion thresholds from docs/design/historical/belief_retention_class.md §4.
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

    Per docs/design/historical/belief_retention_class.md §4 a snapshot is promoted when
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
