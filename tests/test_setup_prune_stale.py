"""Layer A/B/C coverage for #781 — stale-path hook accumulation.

Tests three behaviours that together prevent the ENOENT-spam failure
mode described in issue #781:

* **Layer A — install dedupe.** `install_*_hook` matches existing
  entries by `(matcher, basename)`, so a path change replaces the
  stale entry instead of stacking. Covered as unit tests against
  `install_user_prompt_submit_hook` (representative of every
  matcher-less event) and `install_search_tool_bash_hook` (the
  matcher-keyed surface).
* **Layer B — `aelf setup` prune.** `prune_broken_aelf_hooks` runs
  before the install loop in `_cmd_setup`, sweeping out stale
  entries whose program no longer resolves — but only for
  `aelf-*`-basename entries. Custom shell hooks are left intact.
* **Layer C — `aelf doctor --fix`.** Mirrors Layer B on demand from
  the `doctor` CLI, with a `--dry-run` mode for visibility.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import cast

import pytest

from aelfrice.cli import main
from aelfrice.doctor import prune_broken_aelf_hooks
from aelfrice.setup import (
    install_search_tool_bash_hook,
    install_user_prompt_submit_hook,
)


def _read(path: Path) -> dict[str, object]:
    return cast(dict[str, object], json.loads(path.read_text()))


def _event_list(data: dict[str, object], event: str) -> list[dict[str, object]]:
    hooks = data["hooks"]
    assert isinstance(hooks, dict)
    entries = cast(dict[str, object], hooks)[event]
    assert isinstance(entries, list)
    return cast(list[dict[str, object]], entries)


def _make_live_program(tmp_path: Path, basename: str) -> Path:
    """Drop an executable stub at `tmp_path/basename` and return its abs path."""
    p = tmp_path / basename
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return p


# --- Layer A ---------------------------------------------------------------


def test_install_replaces_basename_match_on_path_change(tmp_path: Path) -> None:
    """A second install with a different absolute path replaces the entry.

    The accumulation bug: `/old/.venv/bin/aelf-hook` followed by
    `/new/.venv/bin/aelf-hook` previously left both entries in place.
    After Layer A, the second call replaces the first.
    """
    settings = tmp_path / "settings.json"
    install_user_prompt_submit_hook(
        settings, command="/old/.venv/bin/aelf-hook",
    )
    install_user_prompt_submit_hook(
        settings, command="/new/.venv/bin/aelf-hook",
    )
    entries = _event_list(_read(settings), "UserPromptSubmit")
    assert len(entries) == 1
    cmd = cast(list[dict[str, object]], entries[0]["hooks"])[0]["command"]
    assert cmd == "/new/.venv/bin/aelf-hook"


def test_install_idempotent_same_command_twice(tmp_path: Path) -> None:
    """Two installs with the same command leave a single entry.

    Regression guard against the dedupe helper accidentally treating
    `entries[i] == new_entry` as a replace+write rather than a no-op.
    """
    settings = tmp_path / "settings.json"
    r1 = install_user_prompt_submit_hook(
        settings, command="/usr/local/bin/aelf-hook",
    )
    r2 = install_user_prompt_submit_hook(
        settings, command="/usr/local/bin/aelf-hook",
    )
    assert r1.installed and not r1.already_present
    assert r2.already_present and not r2.installed
    entries = _event_list(_read(settings), "UserPromptSubmit")
    assert len(entries) == 1


def test_install_replaces_only_within_matcher_partition(tmp_path: Path) -> None:
    """Matcher-keyed install (Bash vs Grep|Glob) does not cross partitions.

    The search-tool hook ships with two PreToolUse entries that share
    `aelf-search-tool-hook` as the basename but live under different
    matchers per the v1.5.0 contract. Layer A's matcher-aware dedupe
    must not collapse them into one.
    """
    settings = tmp_path / "settings.json"
    # Plant the Grep|Glob matcher entry directly so we can verify the
    # Bash install does not steamroll it.
    settings.write_text(json.dumps({
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Grep|Glob",
                    "hooks": [{
                        "type": "command",
                        "command": "/usr/local/bin/aelf-search-tool-hook",
                    }],
                },
            ],
        },
    }))
    install_search_tool_bash_hook(
        settings, command="/different/bin/aelf-search-tool-hook",
    )
    entries = _event_list(_read(settings), "PreToolUse")
    matchers = sorted(cast(str, e.get("matcher", "")) for e in entries)
    assert matchers == ["Bash", "Grep|Glob"]
    assert len(entries) == 2


# --- Layer B / prune helper ------------------------------------------------


def test_prune_collapses_n_stale_to_one_resolvable(tmp_path: Path) -> None:
    """Acceptance: N stale entries + 1 resolvable, prune leaves only the resolvable.

    Mirrors the issue's repro: a settings.json that accumulated five
    aelf-hook entries pointing at venvs that no longer exist, plus one
    pointing at a real binary.
    """
    live = _make_live_program(tmp_path, "aelf-hook")
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": "/missing/a/aelf-hook"}]},
                {"hooks": [{"type": "command", "command": "/missing/b/aelf-hook"}]},
                {"hooks": [{"type": "command", "command": str(live)}]},
                {"hooks": [{"type": "command", "command": "/missing/c/aelf-hook"}]},
                {"hooks": [{"type": "command", "command": "/missing/d/aelf-hook"}]},
                {"hooks": [{"type": "command", "command": "/missing/e/aelf-hook"}]},
            ],
        },
    }))
    result = prune_broken_aelf_hooks(settings)
    assert result.total_removed == 5
    assert result.removed_per_event == {"UserPromptSubmit": 5}
    entries = _event_list(_read(settings), "UserPromptSubmit")
    assert len(entries) == 1
    cmd = cast(list[dict[str, object]], entries[0]["hooks"])[0]["command"]
    assert cmd == str(live)


def test_prune_leaves_custom_shell_hooks_alone(tmp_path: Path) -> None:
    """Acceptance: a custom (non-`aelf-`) broken hook is never touched.

    The prune predicate is intentionally narrow — pruning third-party
    integrations would be a foot-gun. We assert that even a clearly
    broken custom path survives.
    """
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{
                    "type": "command",
                    "command": "/missing/path/aelf-hook",
                }]},
                {"hooks": [{
                    "type": "command",
                    "command": "/missing/path/gh-pii-guard.sh",
                }]},
                {"hooks": [{
                    "type": "command",
                    "command": "/missing/path/conversation-logger.sh",
                }]},
            ],
        },
    }))
    result = prune_broken_aelf_hooks(settings)
    assert result.total_removed == 1
    entries = _event_list(_read(settings), "UserPromptSubmit")
    cmds = sorted(
        cast(str, cast(list[dict[str, object]], e["hooks"])[0]["command"])
        for e in entries
    )
    assert cmds == [
        "/missing/path/conversation-logger.sh",
        "/missing/path/gh-pii-guard.sh",
    ]


def test_prune_dry_run_does_not_rewrite_file(tmp_path: Path) -> None:
    """`dry_run=True` reports the would-be prune but leaves the file intact."""
    settings = tmp_path / "settings.json"
    original = json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{
                    "type": "command",
                    "command": "/missing/aelf-hook",
                }]},
            ],
        },
    }, indent=2)
    settings.write_text(original)
    result = prune_broken_aelf_hooks(settings, dry_run=True)
    assert result.total_removed == 1
    assert settings.read_text() == original


def test_prune_missing_file_is_no_op(tmp_path: Path) -> None:
    """Prune on a non-existent settings.json reports zero and does not create it."""
    settings = tmp_path / "never-existed.json"
    result = prune_broken_aelf_hooks(settings)
    assert result.total_removed == 0
    assert result.removed_per_event == {}
    assert not settings.exists()


def test_prune_skips_shell_metacharacter_commands(tmp_path: Path) -> None:
    """A pipeline / `||` command is `status=skipped`, not `broken`, so prune leaves it.

    Conservative-by-design: the prune predicate only acts on entries
    that doctor would already report `broken`. Anything wrapped in a
    shell pipeline is opaque and left to the user to maintain.
    """
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{
                    "type": "command",
                    "command": "/missing/aelf-hook || true",
                }]},
            ],
        },
    }))
    result = prune_broken_aelf_hooks(settings)
    assert result.total_removed == 0
    entries = _event_list(_read(settings), "UserPromptSubmit")
    assert len(entries) == 1


# --- Layer B in _cmd_setup -------------------------------------------------


def test_cmd_setup_prunes_before_install(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: `aelf setup` sweeps stale aelf-* entries it didn't write.

    Plants a stale `Stop` entry that `aelf setup` is NOT touching
    this run (no `--no-stop-hook` involved, but the stale path is for
    a binary the install will replace anyway). The pre-install prune
    is what catches stale entries the install loop wouldn't otherwise
    rewrite.
    """
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".claude").mkdir()
    settings = proj / ".claude" / "settings.json"
    # Stash a stale aelf-* entry on an event that setup will still
    # reconcile. Two entries here (one resolvable bare-name `aelf-hook`,
    # one missing absolute path) would normally each leak ENOENT.
    settings.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{
                    "type": "command",
                    "command": "/very/missing/.venv/bin/aelf-hook",
                }]},
            ],
            "PreCompact": [
                {"hooks": [{
                    "type": "command",
                    "command": "/very/missing/.venv/bin/aelf-pre-compact-hook",
                }]},
            ],
        },
    }, indent=2))
    # maybe_migrate_to_uv side-effects must not fire during the test —
    # pin its sentinel before invoking setup.
    sentinel_dir = tmp_path / "aelfrice-home"
    sentinel_dir.mkdir()
    monkeypatch.setenv("HOME", str(sentinel_dir))
    (sentinel_dir / ".aelfrice").mkdir()
    (sentinel_dir / ".aelfrice" / "migrated-to-uv").write_text("")
    rc = main([
        "setup",
        "--scope", "project",
        "--project-root", str(proj),
        "--command", "aelf-hook",  # bare-name → resolves via PATH
        "--no-transcript-ingest",
        "--no-session-start",
        "--no-stop-hook",
        "--no-statusline",
        "--no-commit-ingest",
        "--no-search-tool",
        "--no-search-tool-bash",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "pruned" in out
    # Only the freshly installed UserPromptSubmit entry should remain,
    # and the PreCompact slot should be emptied entirely (the prune
    # ran on a stale aelf-pre-compact-hook abs-path that does not
    # exist, and the install loop is not touching PreCompact because
    # --rebuilder was not passed).
    data = _read(settings)
    ups = _event_list(data, "UserPromptSubmit")
    assert len(ups) == 1
    hooks_map = data["hooks"]
    assert isinstance(hooks_map, dict)
    assert cast(dict[str, object], hooks_map).get("PreCompact") == []


# --- Layer C / aelf doctor --fix ------------------------------------------


def test_doctor_fix_prunes_broken_aelf_entries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".claude").mkdir()
    settings = proj / ".claude" / "settings.json"
    settings.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{
                    "type": "command",
                    "command": "/missing/aelf-hook",
                }]},
            ],
        },
    }))
    rc = main([
        "doctor", "hooks",
        "--user-settings", str(tmp_path / "no-such-user-settings.json"),
        "--project-root", str(proj),
        "--fix",
    ])
    # rc reflects pre-fix findings; doctor classified the broken hook
    # before --fix removed it, which is the documented behaviour.
    assert rc == 1
    out = capsys.readouterr().out
    assert "pruned 1 stale aelf-* hook entry" in out
    assert _event_list(_read(settings), "UserPromptSubmit") == []


def test_doctor_fix_dry_run_does_not_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".claude").mkdir()
    settings = proj / ".claude" / "settings.json"
    original = json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{
                    "type": "command",
                    "command": "/missing/aelf-hook",
                }]},
            ],
        },
    }, indent=2) + "\n"
    settings.write_text(original)
    rc = main([
        "doctor", "hooks",
        "--user-settings", str(tmp_path / "no-such-user-settings.json"),
        "--project-root", str(proj),
        "--fix", "--dry-run",
    ])
    assert rc == 1
    out = capsys.readouterr().out
    assert "would prune" in out
    # File untouched byte-for-byte.
    assert settings.read_text() == original


def test_doctor_fix_clean_settings_reports_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".claude").mkdir()
    settings = proj / ".claude" / "settings.json"
    live = _make_live_program(tmp_path, "aelf-hook")
    settings.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": str(live)}]},
            ],
        },
    }))
    rc = main([
        "doctor", "hooks",
        "--user-settings", str(tmp_path / "no-such-user-settings.json"),
        "--project-root", str(proj),
        "--fix",
    ])
    out = capsys.readouterr().out
    assert "no stale aelf-* hook entries to prune" in out
    # Live hook untouched.
    assert len(_event_list(_read(settings), "UserPromptSubmit")) == 1
    # rc may be 0 or 1 depending on other settings.json complaints;
    # the live aelf-hook on its own is not a structural failure.
    _ = rc
