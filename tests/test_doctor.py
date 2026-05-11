"""Tests for `aelf doctor` and the underlying `aelfrice.doctor` module."""
from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import pytest

from aelfrice.cli import main
from aelfrice.doctor import diagnose, format_report


def _write_settings(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _exec(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_diagnose_returns_empty_report_when_no_settings(
    tmp_path: Path,
) -> None:
    report = diagnose(
        user_settings=tmp_path / "missing-user.json",
        project_root=tmp_path / "missing-project",
    )
    assert report.scopes_scanned == []
    assert report.findings == []
    assert "nothing to check" in format_report(report)


def test_diagnose_flags_missing_absolute_path(tmp_path: Path) -> None:
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{
                    "type": "command",
                    "command": "/no/such/aelf-hook",
                }],
            }],
        },
    })
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    broken = report.broken
    assert len(broken) == 1
    assert broken[0].program == "/no/such/aelf-hook"
    assert "does not exist" in broken[0].detail


def test_diagnose_flags_bare_name_not_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "PreToolUse": [{
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": "definitely-not-installed"}],
            }],
        },
    })
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    broken = report.broken
    assert len(broken) == 1
    assert "not on $PATH" in broken[0].detail


def test_diagnose_passes_when_program_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    hook = _exec(bin_dir / "aelf-hook")
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{"type": "command", "command": str(hook)}],
            }],
        },
        "statusLine": {"type": "command", "command": str(hook)},
    })
    monkeypatch.setenv("PATH", str(bin_dir))
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert report.broken == []
    assert report.ok_count == 2


def test_diagnose_skips_shell_pipe_commands(tmp_path: Path) -> None:
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "PreToolUse": [{
                "hooks": [{
                    "type": "command",
                    "command": "true | echo skipped && exit 0",
                }],
            }],
        },
    })
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert report.broken == []
    assert report.skipped_count == 1


def test_diagnose_inspects_script_under_interpreter(
    tmp_path: Path,
) -> None:
    user_path = tmp_path / "settings.json"
    # `bash <missing-script>` should report the SCRIPT, not bash.
    _write_settings(user_path, {
        "hooks": {
            "PreToolUse": [{
                "hooks": [{
                    "type": "command",
                    "command": f"bash {tmp_path / 'gone.sh'}",
                }],
            }],
        },
    })
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert len(report.broken) == 1
    assert "gone.sh" in report.broken[0].program


def test_diagnose_inspects_script_through_silent_failure_wrapper(
    tmp_path: Path,
) -> None:
    """Issue #113: a stale `bash <missing>.sh 2>/dev/null || true`
    hook silently skipped past doctor because the shell-meta check
    short-circuited before we extracted the script path. We now
    extract the script even when the wrapper appears."""
    user_path = tmp_path / "settings.json"
    missing = tmp_path / "hook-aelf-search.sh"
    _write_settings(user_path, {
        "hooks": {
            "PostToolUse": [{
                "matcher": "Grep|Glob",
                "hooks": [{
                    "type": "command",
                    "command": f"bash {missing} 2>/dev/null || true",
                    "timeout": 5,
                }],
            }],
        },
    })
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert len(report.broken) == 1, report.findings
    assert "hook-aelf-search.sh" in report.broken[0].program
    assert report.broken[0].silent_failure is True


def test_diagnose_flags_silent_failure_pattern_on_resolved_script(
    tmp_path: Path,
) -> None:
    """Even when the script exists, the wrapper itself is the
    anti-feature (issue #114). Surface a soft warning."""
    bin_dir = tmp_path / "bin"
    hook = _exec(bin_dir / "ok-script.sh")
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "PostToolUse": [{
                "hooks": [{
                    "type": "command",
                    "command": f"bash {hook} 2>/dev/null || true",
                }],
            }],
        },
    })
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert report.broken == []
    assert len(report.silent_failure) == 1
    rendered = format_report(report)
    assert "silent-failure pattern" in rendered


def test_diagnose_surfaces_recent_hook_failures(tmp_path: Path) -> None:
    """When ~/.aelfrice/logs/hook-failures.log exists, doctor tails
    it into the report (issue #114)."""
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {"hooks": {}})
    log = tmp_path / "hook-failures.log"
    log.write_text(
        "\n".join([
            "2026-04-27T22:00:00 hook-aelf-search: file not found",
            "2026-04-27T22:01:00 hook-aelf-search: file not found",
            "",
        ]),
        encoding="utf-8",
    )
    report = diagnose(
        user_settings=user_path,
        project_root=tmp_path / "noproj",
        hook_failures_log=log,
    )
    assert len(report.hook_failures_tail) == 2
    rendered = format_report(report)
    assert "recent hook failures" in rendered
    assert "hook-aelf-search" in rendered


def test_diagnose_no_log_no_section(tmp_path: Path) -> None:
    """Missing log → no log block in the report (no false noise)."""
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {"hooks": {}})
    report = diagnose(
        user_settings=user_path,
        project_root=tmp_path / "noproj",
        hook_failures_log=tmp_path / "no-log.log",
    )
    assert report.hook_failures_tail == ()
    rendered = format_report(report)
    assert "recent hook failures" not in rendered


def test_doctor_cli_exit_1_when_broken(tmp_path: Path) -> None:
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{"type": "command", "command": "/no/such"}],
            }],
        },
    })
    buf = io.StringIO()
    code = main(
        argv=[
            "doctor",
            "--user-settings", str(user_path),
            "--project-root", str(tmp_path / "noproj"),
        ],
        out=buf,
    )
    assert code == 1
    assert "broken" in buf.getvalue()


def test_diagnose_flags_orphan_slash_command(tmp_path: Path) -> None:
    """A slash file naming a subcommand the CLI doesn't have is an
    orphan (issue #115)."""
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {"hooks": {}})
    slash_dir = tmp_path / "commands" / "aelf"
    slash_dir.mkdir(parents=True)
    (slash_dir / "ingest-transcript.md").write_text("# slash", encoding="utf-8")
    (slash_dir / "search.md").write_text("# slash", encoding="utf-8")
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj",
        slash_commands_dir=slash_dir,
        known_cli_subcommands=frozenset({"search"}),
    )
    assert report.orphan_slash_commands == ["ingest-transcript"]
    rendered = format_report(report)
    assert "/aelf:ingest-transcript" in rendered
    assert "no `aelf ingest-transcript` subcommand" in rendered


def test_diagnose_no_orphan_slash_when_all_match(tmp_path: Path) -> None:
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {"hooks": {}})
    slash_dir = tmp_path / "commands" / "aelf"
    slash_dir.mkdir(parents=True)
    (slash_dir / "search.md").write_text("# slash", encoding="utf-8")
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj",
        slash_commands_dir=slash_dir,
        known_cli_subcommands=frozenset({"search", "stats"}),
    )
    assert report.orphan_slash_commands == []


def test_diagnose_skips_slash_check_when_known_not_provided(
    tmp_path: Path,
) -> None:
    """Backwards-compat: doctor's settings linter still runs without
    being asked to verify slash commands."""
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {"hooks": {}})
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj",
    )
    assert report.orphan_slash_commands == []


def test_doctor_cli_exit_0_when_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    hook = _exec(bin_dir / "aelf-hook")
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{"type": "command", "command": str(hook)}],
            }],
        },
    })
    monkeypatch.setenv("PATH", str(bin_dir))
    buf = io.StringIO()
    code = main(
        argv=[
            "doctor",
            "--user-settings", str(user_path),
            "--project-root", str(tmp_path / "noproj"),
        ],
        out=buf,
    )
    assert code == 0
    assert "1 ok" in buf.getvalue()


# --- v2.1 default-on auto-capture nag (#557) -----------------------------


def test_diagnose_flags_missing_auto_capture_hooks_when_pre_v21(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-v2.1 install: settings has only the retrieval hook, none of
    the three default-on auto-capture hooks. Report should list all
    three as missing and the rendered output should print the re-run
    nag."""
    bin_dir = tmp_path / "bin"
    retrieval_hook = _exec(bin_dir / "aelf-hook")
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{
                "hooks": [{"type": "command", "command": str(retrieval_hook)}],
            }],
        },
    })
    monkeypatch.setenv("PATH", str(bin_dir))
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert report.missing_auto_capture_hooks == [
        "aelf-transcript-logger",
        "aelf-commit-ingest",
        "aelf-session-start-hook",
        "aelf-stop-hook",
    ]
    rendered = format_report(report)
    assert "auto-capture hooks not installed" in rendered
    assert "re-run 'aelf setup'" in rendered


def test_diagnose_quiet_when_all_auto_capture_hooks_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fresh v2.1+ install: all default-on hooks present (including the
    #582 stop hook). No nag line in the rendered report."""
    bin_dir = tmp_path / "bin"
    retrieval = _exec(bin_dir / "aelf-hook")
    transcript = _exec(bin_dir / "aelf-transcript-logger")
    commit_ingest = _exec(bin_dir / "aelf-commit-ingest")
    session_start = _exec(bin_dir / "aelf-session-start-hook")
    stop_hook = _exec(bin_dir / "aelf-stop-hook")
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": str(retrieval)}]},
                {"hooks": [{"type": "command", "command": str(transcript)}]},
            ],
            "Stop": [
                {"hooks": [{"type": "command", "command": str(transcript)}]},
                {"hooks": [{"type": "command", "command": str(stop_hook)}]},
            ],
            "PreCompact": [
                {"hooks": [{"type": "command", "command": str(transcript)}]},
            ],
            "PostCompact": [
                {"hooks": [{"type": "command", "command": str(transcript)}]},
            ],
            "PostToolUse": [{
                "matcher": "Bash",
                "hooks": [{"type": "command", "command": str(commit_ingest)}],
            }],
            "SessionStart": [
                {"hooks": [{"type": "command", "command": str(session_start)}]},
            ],
        },
    })
    monkeypatch.setenv("PATH", str(bin_dir))
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert report.missing_auto_capture_hooks == []
    rendered = format_report(report)
    assert "auto-capture hooks not installed" not in rendered


def test_diagnose_partial_auto_capture_lists_only_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixed install: transcript-logger present, the other two absent.
    Only the absent two should appear in the missing list."""
    bin_dir = tmp_path / "bin"
    transcript = _exec(bin_dir / "aelf-transcript-logger")
    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": str(transcript)}]},
            ],
        },
    })
    monkeypatch.setenv("PATH", str(bin_dir))
    report = diagnose(
        user_settings=user_path, project_root=tmp_path / "noproj"
    )
    assert report.missing_auto_capture_hooks == [
        "aelf-commit-ingest",
        "aelf-session-start-hook",
        "aelf-stop-hook",
    ]


def test_diagnose_quiet_when_no_settings_scanned(tmp_path: Path) -> None:
    """No settings.json at either scope → suppress the auto-capture
    nag (the no-scopes-scanned message already covers that case)."""
    report = diagnose(
        user_settings=tmp_path / "missing-user.json",
        project_root=tmp_path / "missing-project",
    )
    rendered = format_report(report)
    assert "auto-capture hooks not installed" not in rendered


def test_auto_capture_basenames_match_setup() -> None:
    """Guardrail: doctor's hardcoded auto-capture basename list must
    stay in sync with `aelfrice.setup`. If setup renames a script,
    this test fires before the doctor check silently misses it."""
    from aelfrice import setup
    from aelfrice.doctor import _AUTO_CAPTURE_HOOK_BASENAMES
    assert set(_AUTO_CAPTURE_HOOK_BASENAMES) == {
        setup.TRANSCRIPT_LOGGER_SCRIPT_NAME,
        setup.COMMIT_INGEST_SCRIPT_NAME,
        setup.SESSION_START_HOOK_SCRIPT_NAME,
        setup.STOP_HOOK_SCRIPT_NAME,
    }


# ---------------------------------------------------------------------------
# #589 — per-project legacy-schema detection
# ---------------------------------------------------------------------------

def _make_legacy_db(path: Path, belief_count: int = 3) -> None:
    """Create a per-project memory.db on the pre-v1.x schema (no `origin`).

    Carries all columns that `migrate._read_legacy_beliefs` reads back
    so an auto-migrate pass over this fixture round-trips through
    `MemoryStore`. The shape matches what real pre-v1.x aelfrice DBs
    looked like in the field — minus the `origin` column added in v1.x.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE beliefs (
            id                TEXT PRIMARY KEY,
            content           TEXT NOT NULL,
            content_hash      TEXT NOT NULL,
            alpha             REAL NOT NULL,
            beta              REAL NOT NULL,
            type              TEXT NOT NULL,
            lock_level        TEXT NOT NULL,
            locked_at         TEXT,
            demotion_pressure INTEGER NOT NULL DEFAULT 0,
            created_at        TEXT NOT NULL,
            last_retrieved_at TEXT
        );
        CREATE TABLE edges (
            src    TEXT NOT NULL,
            dst    TEXT NOT NULL,
            type   TEXT NOT NULL,
            weight REAL NOT NULL,
            PRIMARY KEY (src, dst, type)
        );
        """
    )
    for i in range(belief_count):
        con.execute(
            "INSERT INTO beliefs "
            "(id, content, content_hash, alpha, beta, type, lock_level, "
            " demotion_pressure, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                f"belief-{i:06d}",
                f"belief {i} content",
                f"hash{i:08d}",
                1.0,
                1.0,
                "fact",
                "none",
                0,
                "2026-01-01T00:00:00Z",
            ),
        )
    con.commit()
    con.close()


def _make_modern_db(path: Path, belief_count: int = 2) -> None:
    """Create a per-project memory.db WITH the `origin` column (modern schema)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE beliefs "
        "(id INTEGER PRIMARY KEY, content TEXT, type TEXT, origin TEXT)"
    )
    for i in range(belief_count):
        con.execute(
            "INSERT INTO beliefs (content, type, origin) VALUES (?, ?, ?)",
            (f"belief {i}", "fact", "agent_remembered"),
        )
    con.commit()
    con.close()


def _make_empty_legacy_db(path: Path) -> None:
    """Create a per-project memory.db without `origin`, but with zero rows (skip)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE beliefs "
        "(id INTEGER PRIMARY KEY, content TEXT, type TEXT)"
    )
    con.commit()
    con.close()


def test_legacy_schema_detected(tmp_path: Path) -> None:
    """Legacy DB (no `origin` column, rows present) → block present in report;
    modern DB with `origin` column → absent from block. Also verifies that
    an empty legacy DB (zero rows) is silently skipped (#589 AC).
    """
    from aelfrice.doctor import _check_legacy_schema_dbs

    projects_dir = tmp_path / "projects"

    # Legacy DB with rows — should be flagged.
    legacy_path = projects_dir / "aabbccdd1234" / "memory.db"
    _make_legacy_db(legacy_path, belief_count=5)

    # Modern DB with origin column — should be skipped.
    modern_path = projects_dir / "11223344aabb" / "memory.db"
    _make_modern_db(modern_path, belief_count=2)

    # Empty legacy DB (no rows) — should be skipped.
    empty_legacy_path = projects_dir / "deadbeef0000" / "memory.db"
    _make_empty_legacy_db(empty_legacy_path)

    results = _check_legacy_schema_dbs(projects_dir=projects_dir)

    paths_found = [r.path for r in results]
    assert legacy_path in paths_found, "legacy DB with rows must be flagged"
    assert modern_path not in paths_found, "modern DB must not be flagged"
    assert empty_legacy_path not in paths_found, "empty legacy DB must not be flagged"

    assert len(results) == 1
    entry = results[0]
    assert entry.row_count == 5
    assert entry.idle_days >= 0


def test_legacy_schema_auto_migrate_success(tmp_path: Path) -> None:
    """`diagnose` auto-migrates legacy DBs in place (#593).

    Verifies the operator-decision contract:
      * The pre-v1.x file is preserved at `<path>.pre-v1x.bak`.
      * A modern-schema DB lives at the original path (has `origin` col).
      * `report.migrated_dbs` carries one entry with the row count and
        a non-negative duration.
      * `report.failed_migrate_dbs` is empty.
      * The format pass emits a `migrated <path>: <N> beliefs, ...`
        summary line — no more `legacy-schema per-project DBs detected`
        nag, no more `aelf migrate --from` fix instruction.
    """
    projects_dir = tmp_path / "projects"
    legacy_path = projects_dir / "abc123def456" / "memory.db"
    _make_legacy_db(legacy_path, belief_count=7)

    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "aelf-hook"}]}],
        },
    })
    report = diagnose(
        user_settings=user_path,
        project_root=tmp_path / "noproj",
        aelfrice_projects_dir=projects_dir,
    )

    # Pre-migration detection set is preserved on the report.
    assert len(report.legacy_schema_dbs) == 1

    # Auto-migrate ran successfully.
    assert len(report.migrated_dbs) == 1
    assert len(report.failed_migrate_dbs) == 0
    migrated = report.migrated_dbs[0]
    assert migrated.path == legacy_path
    assert migrated.backup_path == legacy_path.with_name(
        legacy_path.name + ".pre-v1x.bak"
    )
    assert migrated.row_count == 7
    assert migrated.duration_ms >= 0

    # Files: backup preserved, modern-schema DB at original path.
    assert migrated.backup_path.is_file(), "backup must be preserved"
    assert legacy_path.is_file(), "modern-schema DB must exist at original path"

    # The DB at original path now carries `origin`.
    con = sqlite3.connect(f"file:{legacy_path}?mode=ro", uri=True)
    try:
        cols = {row[1] for row in con.execute("PRAGMA table_info(beliefs)")}
    finally:
        con.close()
    assert "origin" in cols, "post-migration DB must have the origin column"

    # Format pass: summary line present, old nag absent.
    rendered = format_report(report)
    assert f"migrated {legacy_path}" in rendered
    assert "7 beliefs" in rendered
    assert str(migrated.backup_path) in rendered
    assert "legacy-schema per-project DBs detected" not in rendered
    assert "aelf migrate --from" not in rendered


def test_legacy_schema_auto_migrate_failure_when_backup_exists(
    tmp_path: Path,
) -> None:
    """A stale `.pre-v1x.bak` blocks auto-migrate (#593 recoverability guard).

    `migrate_in_place` refuses to overwrite an existing backup so the
    operator can recover from prior failed runs. `diagnose()` surfaces
    the failure via `failed_migrate_dbs` and renders the residual nag
    in format_report.
    """
    projects_dir = tmp_path / "projects"
    legacy_path = projects_dir / "deadbeefcafe" / "memory.db"
    _make_legacy_db(legacy_path, belief_count=3)

    # Pre-existing backup blocks the migration.
    stale_backup = legacy_path.with_name(legacy_path.name + ".pre-v1x.bak")
    stale_backup.write_bytes(b"prior-run-artifact")

    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "aelf-hook"}]}],
        },
    })
    report = diagnose(
        user_settings=user_path,
        project_root=tmp_path / "noproj",
        aelfrice_projects_dir=projects_dir,
    )

    assert len(report.migrated_dbs) == 0
    assert len(report.failed_migrate_dbs) == 1
    failure = report.failed_migrate_dbs[0]
    assert failure.path == legacy_path
    assert failure.reason == "FileExistsError"

    # Legacy file untouched on failure; stale backup byte-identical.
    assert legacy_path.is_file()
    assert stale_backup.read_bytes() == b"prior-run-artifact"

    rendered = format_report(report)
    assert "auto-migrate FAILED" in rendered
    assert str(legacy_path) in rendered
    assert "FileExistsError" in rendered
    assert "aelf migrate --from" in rendered


def test_migrate_in_place_round_trip(tmp_path: Path) -> None:
    """`migrate_in_place` direct-call: backup made, content preserved (#593).

    Bypasses `diagnose()` so a regression in the doctor-side orchestrator
    can't mask a regression in the schema-mutation primitive itself.
    """
    from aelfrice.migrate import migrate_in_place

    db_path = tmp_path / "memory.db"
    _make_legacy_db(db_path, belief_count=4)

    report = migrate_in_place(db_path)

    assert report.db_path == db_path
    assert report.backup_path == db_path.with_name(db_path.name + ".pre-v1x.bak")
    assert report.counts.inserted_beliefs == 4
    assert report.duration_ms >= 0
    assert report.backup_path.is_file()
    assert db_path.is_file()

    # Post-migration DB has the modern schema's origin column.
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cols = {row[1] for row in con.execute("PRAGMA table_info(beliefs)")}
    finally:
        con.close()
    assert "origin" in cols


def test_migrate_in_place_refuses_existing_backup(tmp_path: Path) -> None:
    """`migrate_in_place` raises FileExistsError when a backup is already present.

    Keeps prior-run artifacts recoverable instead of clobbering them.
    """
    from aelfrice.migrate import migrate_in_place

    db_path = tmp_path / "memory.db"
    _make_legacy_db(db_path, belief_count=1)
    db_path.with_name(db_path.name + ".pre-v1x.bak").write_bytes(b"prior")

    with pytest.raises(FileExistsError):
        migrate_in_place(db_path)


def test_migrate_in_place_raises_when_missing(tmp_path: Path) -> None:
    """`migrate_in_place` raises FileNotFoundError when the input is absent."""
    from aelfrice.migrate import migrate_in_place

    with pytest.raises(FileNotFoundError):
        migrate_in_place(tmp_path / "does-not-exist" / "memory.db")


def test_legacy_schema_report_quiet_when_zero(tmp_path: Path) -> None:
    """When no legacy DBs exist, format_report must not emit the nag block."""
    projects_dir = tmp_path / "projects"
    modern_path = projects_dir / "00112233aabb" / "memory.db"
    _make_modern_db(modern_path, belief_count=3)

    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "aelf-hook"}]}],
        },
    })
    report = diagnose(
        user_settings=user_path,
        project_root=tmp_path / "noproj",
        aelfrice_projects_dir=projects_dir,
    )
    assert report.legacy_schema_dbs == []
    rendered = format_report(report)
    assert "legacy-schema per-project DBs detected" not in rendered


def test_legacy_schema_quiet_when_projects_dir_missing(tmp_path: Path) -> None:
    """Non-existent projects dir → no crash, empty list."""
    from aelfrice.doctor import _check_legacy_schema_dbs

    results = _check_legacy_schema_dbs(
        projects_dir=tmp_path / "no-such-projects-dir"
    )
    assert results == []


def test_legacy_schema_idle_days_in_report(tmp_path: Path) -> None:
    """The rendered block includes 'idle Xd' for the legacy DB."""
    projects_dir = tmp_path / "projects"
    legacy_path = projects_dir / "ffee12345678" / "memory.db"
    _make_legacy_db(legacy_path, belief_count=4)

    user_path = tmp_path / "settings.json"
    _write_settings(user_path, {
        "hooks": {
            "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "aelf-hook"}]}],
        },
    })
    report = diagnose(
        user_settings=user_path,
        project_root=tmp_path / "noproj",
        aelfrice_projects_dir=projects_dir,
    )
    rendered = format_report(report)
    assert "idle" in rendered
    assert "beliefs" in rendered
