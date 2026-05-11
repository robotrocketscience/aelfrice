"""Tests for `aelfrice.auto_install` — version-stamped manifest merger.

Covers the #623 acceptance bullets:
- Manifest loads and lists the expected default-on hooks.
- First run (no stamp) creates the dotfile + merges all hooks into
  ~/.claude/settings.json and prints one stderr line.
- Stamped == installed → fast no-op (no settings.json mutation).
- Stamped < installed → updates delta only, leaves preserved entries
  intact, prints `hooks updated to vX.Y.Z (was vA.B.C)`.
- Opt-out file persists across upgrades (the opted-out hook is NOT
  re-added).
- User-added unrelated entries preserved byte-identical.
- AELFRICE_NO_AUTO_INSTALL=1 skips the whole flow.
- Failure leaves stamp untouched (next invocation retries).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aelfrice import auto_install
from aelfrice.auto_install import (
    AutoInstallResult,
    _UNSTAMPED,
    add_opt_out,
    is_disabled_via_env,
    load_manifest,
    maybe_install_manifest,
    read_opt_outs,
    read_stamp,
    remove_opt_out,
    write_stamp,
)


# --- manifest loader ----------------------------------------------------


def test_load_manifest_returns_known_defaults() -> None:
    m = load_manifest()
    assert m.schema_version == 1
    names = [h.name for h in m.hooks]
    assert "user_prompt_submit_retrieval" in names
    assert "transcript_ingest" in names
    assert "commit_ingest" in names
    assert "session_start" in names
    assert "stop_lock_prompt" in names
    # All bundled hooks ship default-on.
    assert all(h.default_on for h in m.hooks)


def test_load_manifest_owned_basenames_covers_setup_surface() -> None:
    m = load_manifest()
    owned = m.owned_basenames()
    assert "aelf-hook" in owned
    assert "aelf-transcript-logger" in owned
    assert "aelf-commit-ingest" in owned
    assert "aelf-session-start-hook" in owned
    assert "aelf-stop-hook" in owned


# --- stamp file ---------------------------------------------------------


def test_read_stamp_missing_returns_unstamped(tmp_path: Path) -> None:
    assert read_stamp(tmp_path / "no-such-stamp") == _UNSTAMPED


def test_read_stamp_strips_trailing_newline(tmp_path: Path) -> None:
    stamp = tmp_path / "stamp"
    stamp.write_text("2.1.0\n", encoding="utf-8")
    assert read_stamp(stamp) == "2.1.0"


def test_write_stamp_atomic_and_readable(tmp_path: Path) -> None:
    stamp = tmp_path / "nested" / "stamp"
    write_stamp(stamp, "2.2.0")
    assert read_stamp(stamp) == "2.2.0"


# --- opt-out file -------------------------------------------------------


def test_read_opt_outs_missing_returns_empty(tmp_path: Path) -> None:
    assert read_opt_outs(tmp_path / "no-such-file") == frozenset()


def test_add_and_remove_opt_out_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "opt-out-hooks.json"
    add_opt_out("transcript_ingest", p)
    assert read_opt_outs(p) == frozenset({"transcript_ingest"})
    add_opt_out("commit_ingest", p)
    assert read_opt_outs(p) == frozenset({"transcript_ingest", "commit_ingest"})
    # Idempotent re-add.
    add_opt_out("commit_ingest", p)
    assert read_opt_outs(p) == frozenset({"transcript_ingest", "commit_ingest"})
    remove_opt_out("commit_ingest", p)
    assert read_opt_outs(p) == frozenset({"transcript_ingest"})
    # Removing the last entry deletes the file.
    remove_opt_out("transcript_ingest", p)
    assert not p.exists()


def test_read_opt_outs_malformed_returns_empty(tmp_path: Path) -> None:
    p = tmp_path / "opt-out-hooks.json"
    p.write_text("not-json", encoding="utf-8")
    assert read_opt_outs(p) == frozenset()


# --- maybe_install_manifest happy paths ---------------------------------


def _settings_setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Return (settings_path, stamp_path, opt_out_path) all under tmp_path."""
    return (
        tmp_path / "claude-settings.json",
        tmp_path / "stamp",
        tmp_path / "opt-out-hooks.json",
    )


def test_first_run_creates_stamp_and_writes_all_hooks(tmp_path: Path) -> None:
    settings, stamp, opt_out = _settings_setup(tmp_path)
    result = maybe_install_manifest(
        installed_version="2.2.0",
        scope="user",
        settings_path=settings,
        stamp_path=stamp,
        opt_out_path=opt_out,
    )
    assert result.ran
    assert result.prev_version == _UNSTAMPED
    assert result.new_version == "2.2.0"
    assert set(result.installed) >= {
        "user_prompt_submit_retrieval",
        "transcript_ingest",
        "commit_ingest",
        "session_start",
        "stop_lock_prompt",
    }
    assert read_stamp(stamp) == "2.2.0"
    # The first-run message mentions "installed default hooks".
    assert "installed default hooks" in result.message
    # settings.json now has all the expected event keys.
    data = json.loads(settings.read_text(encoding="utf-8"))
    hooks_block = data["hooks"]
    assert "UserPromptSubmit" in hooks_block
    assert "Stop" in hooks_block
    assert "PreCompact" in hooks_block
    assert "PostCompact" in hooks_block
    assert "PostToolUse" in hooks_block  # commit-ingest
    assert "SessionStart" in hooks_block


def test_stamp_equal_to_installed_is_fast_no_op(tmp_path: Path) -> None:
    settings, stamp, opt_out = _settings_setup(tmp_path)
    write_stamp(stamp, "2.2.0")
    # No settings.json on disk and no install_fn call is made — the gate
    # short-circuits before _do_merge runs. If it did run we would crash
    # trying to write under a non-existent claude dir, which is fine here
    # because settings_path is under tmp_path.
    result = maybe_install_manifest(
        installed_version="2.2.0",
        settings_path=settings,
        stamp_path=stamp,
        opt_out_path=opt_out,
    )
    assert result.ran is False
    assert result.installed == ()
    assert result.message == ""
    # settings.json was never created.
    assert not settings.exists()


def test_upgrade_delta_message(tmp_path: Path) -> None:
    settings, stamp, opt_out = _settings_setup(tmp_path)
    # Simulate previously merged at v2.0.0.
    write_stamp(stamp, "2.0.0")
    # And a pre-existing settings.json with only the v2.0.0 hooks present
    # by running an install at the old "version" first.
    maybe_install_manifest(
        installed_version="2.0.0",
        settings_path=settings,
        stamp_path=stamp,
        opt_out_path=opt_out,
        force=True,
    )
    # Now "upgrade" to v2.2.0. The hooks are already there, so the merge
    # is a no-op at the entry level (all "already"), but the stamp bumps
    # and the message reflects the version bump only when entries change.
    result = maybe_install_manifest(
        installed_version="2.2.0",
        settings_path=settings,
        stamp_path=stamp,
        opt_out_path=opt_out,
    )
    assert result.ran is True
    assert result.prev_version == "2.0.0"
    assert result.new_version == "2.2.0"
    # All hooks already present from the prior merge -> no fresh installs.
    assert result.installed == ()
    assert set(result.already) >= {"transcript_ingest", "commit_ingest"}
    # No "added" line when nothing was actually added.
    assert result.message == ""
    # But the stamp bumps anyway so the next invocation is a fast no-op.
    assert read_stamp(stamp) == "2.2.0"


def test_upgrade_with_new_hook_writes_only_the_delta(tmp_path: Path) -> None:
    """Simulate a fresh hook entering the manifest between releases.

    We can't easily mutate the bundled manifest in a test, so instead we
    drop one hook by writing a fake stamp + a settings.json that only
    contains 4 of 5 hooks, then re-run the merge. The 5th hook should
    appear in `installed`.
    """
    settings, stamp, opt_out = _settings_setup(tmp_path)
    # First run: install everything at v2.1.0.
    maybe_install_manifest(
        installed_version="2.1.0",
        settings_path=settings,
        stamp_path=stamp,
        opt_out_path=opt_out,
    )
    # Surgically strip the Stop hook entry so the next merge re-adds it.
    data = json.loads(settings.read_text(encoding="utf-8"))
    data["hooks"]["Stop"] = [
        e for e in data["hooks"]["Stop"]
        if "aelf-stop-hook" not in str(e)
    ]
    settings.write_text(json.dumps(data), encoding="utf-8")
    # Now "upgrade" to v2.2.0.
    result = maybe_install_manifest(
        installed_version="2.2.0",
        settings_path=settings,
        stamp_path=stamp,
        opt_out_path=opt_out,
    )
    assert result.ran is True
    assert "stop_lock_prompt" in result.installed
    assert "hooks updated to v2.2.0 (was v2.1.0)" in result.message


# --- opt-out interaction -------------------------------------------------


def test_opt_out_skips_named_hook_across_upgrade(tmp_path: Path) -> None:
    settings, stamp, opt_out = _settings_setup(tmp_path)
    add_opt_out("commit_ingest", opt_out)
    result = maybe_install_manifest(
        installed_version="2.2.0",
        settings_path=settings,
        stamp_path=stamp,
        opt_out_path=opt_out,
    )
    assert result.ran is True
    assert "commit_ingest" in result.opted_out
    assert "commit_ingest" not in result.installed
    # settings.json must not have the commit-ingest PostToolUse entry.
    data = json.loads(settings.read_text(encoding="utf-8"))
    post_tool_use = data["hooks"].get("PostToolUse", [])
    assert not any(
        "aelf-commit-ingest" in str(entry) for entry in post_tool_use
    )


# --- user-added entries preserved ---------------------------------------


def test_unrelated_user_entries_preserved(tmp_path: Path) -> None:
    settings, stamp, opt_out = _settings_setup(tmp_path)
    # Pre-existing settings.json with the user's own hook + unrelated keys.
    settings.write_text(
        json.dumps({
            "permissions": {"allow": ["WebSearch"]},
            "hooks": {
                "UserPromptSubmit": [
                    {"hooks": [{
                        "type": "command",
                        "command": "/usr/local/bin/my-custom-hook",
                    }]}
                ]
            },
            "model": "test-model-id",
        }),
        encoding="utf-8",
    )
    maybe_install_manifest(
        installed_version="2.2.0",
        settings_path=settings,
        stamp_path=stamp,
        opt_out_path=opt_out,
    )
    data = json.loads(settings.read_text(encoding="utf-8"))
    # Unrelated top-level keys survived.
    assert data["permissions"] == {"allow": ["WebSearch"]}
    assert data["model"] == "test-model-id"
    # The user's UserPromptSubmit entry is still there.
    ups_entries = data["hooks"]["UserPromptSubmit"]
    user_cmd_present = any(
        "/usr/local/bin/my-custom-hook" in str(e) for e in ups_entries
    )
    assert user_cmd_present
    # And our aelf-hook was added alongside it.
    aelf_cmd_present = any(
        "aelf-hook" in str(e) for e in ups_entries
    )
    assert aelf_cmd_present


# --- env-var bypass ------------------------------------------------------


def test_env_var_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_NO_AUTO_INSTALL", "1")
    assert is_disabled_via_env() is True


def test_env_var_unset_does_not_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AELFRICE_NO_AUTO_INSTALL", raising=False)
    assert is_disabled_via_env() is False


def test_env_var_empty_does_not_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_NO_AUTO_INSTALL", "")
    assert is_disabled_via_env() is False


def test_auto_install_at_cli_entry_bypasses_when_env_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """When AELFRICE_NO_AUTO_INSTALL=1, the helper must not even read the manifest."""
    monkeypatch.setenv("AELFRICE_NO_AUTO_INSTALL", "1")
    # Point STAMP_PATH at tmp so a successful run would create files we can detect.
    monkeypatch.setattr(auto_install, "STAMP_PATH", tmp_path / "stamp")
    monkeypatch.setattr(auto_install, "OPT_OUT_PATH", tmp_path / "opt-out")
    monkeypatch.setattr(auto_install, "USER_SETTINGS_PATH", tmp_path / "settings.json")
    auto_install.auto_install_at_cli_entry(installed_version="2.2.0")
    # Nothing written, nothing printed.
    assert not (tmp_path / "stamp").exists()
    captured = capsys.readouterr()
    assert captured.err == ""


# --- failure semantics ---------------------------------------------------


def test_failure_during_install_leaves_stamp_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    settings, stamp, opt_out = _settings_setup(tmp_path)
    write_stamp(stamp, "2.0.0")
    # Force one of the install functions to blow up.
    def boom(*_a: object, **_kw: object) -> object:
        raise RuntimeError("synthetic")
    monkeypatch.setattr(
        auto_install, "install_session_start_hook", boom
    )
    # Reset dispatch table for this hook to point at the broken fn.
    monkeypatch.setitem(
        auto_install._DISPATCH,
        "session_start",
        (auto_install.resolve_session_start_hook_command, boom),
    )
    with pytest.raises(RuntimeError, match="synthetic"):
        maybe_install_manifest(
            installed_version="2.2.0",
            settings_path=settings,
            stamp_path=stamp,
            opt_out_path=opt_out,
        )
    # Stamp must NOT have advanced.
    assert read_stamp(stamp) == "2.0.0"
