"""Codex host tooling: hooks.json write / doctor / removal (#1052).

All paths are tmp_path-scoped; nothing touches a real ~/.codex. The
"broken hooks.json" cases mirror a real-world observation: a 14-byte
truncated file that any naive read-modify-write would clobber.
"""
from __future__ import annotations

import json
from pathlib import Path

from aelfrice.host_codex import (
    CodexDoctorReport,
    desired_codex_hooks,
    doctor_codex,
    install_codex_hooks,
    remove_codex_hooks,
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


# --- desired set shape -----------------------------------------------------


def test_desired_set_covers_portable_events() -> None:
    events = set(desired_codex_hooks().keys())
    assert events == {
        "UserPromptSubmit", "Stop", "PreCompact", "PostCompact",
        "SessionStart", "PreToolUse", "PostToolUse",
    }


def test_tool_hooks_match_bash_only() -> None:
    """#1055: Codex canonicalizes shell commands to tool_name "Bash";
    Grep/Glob tools do not exist there and must not be matched."""
    desired = desired_codex_hooks()
    for event in ("PreToolUse", "PostToolUse"):
        for group in desired[event]:
            assert group["matcher"] == "Bash"
    pre_cmds = [
        h["command"] for g in desired["PreToolUse"] for h in g["hooks"]
    ]
    assert any("aelf-search-tool-hook" in c for c in pre_cmds)
    assert any("aelf-pre-issue-hook" in c for c in pre_cmds)
    post_cmds = [
        h["command"] for g in desired["PostToolUse"] for h in g["hooks"]
    ]
    assert any("aelf-commit-ingest" in c for c in post_cmds)
    # The Claude-memory mirror is host-specific and must stay out.
    all_cmds = pre_cmds + post_cmds
    assert not any("claude-memory-mirror" in c for c in all_cmds)


def test_session_start_matcher_includes_compact() -> None:
    groups = desired_codex_hooks()["SessionStart"]
    assert "compact" in str(groups[0]["matcher"])


# --- install ---------------------------------------------------------------


def test_install_creates_file_with_all_events(tmp_path: Path) -> None:
    path = tmp_path / "hooks.json"
    result = install_codex_hooks(path)
    assert result.error is None
    assert result.changed
    doc = _read(path)
    assert set(doc["hooks"].keys()) == set(desired_codex_hooks().keys())


def test_install_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "hooks.json"
    install_codex_hooks(path)
    first = path.read_text(encoding="utf-8")
    second_result = install_codex_hooks(path)
    assert not second_result.changed
    assert path.read_text(encoding="utf-8") == first


def test_install_preserves_foreign_entries(tmp_path: Path) -> None:
    path = tmp_path / "hooks.json"
    foreign = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "/usr/local/bin/other-tool"}]},
            ],
            "PreToolUse": [
                {"matcher": "shell", "hooks": [
                    {"type": "command", "command": "/opt/guard.sh"},
                ]},
            ],
        },
    }
    path.write_text(json.dumps(foreign), encoding="utf-8")
    result = install_codex_hooks(path)
    assert result.error is None
    doc = _read(path)
    session_cmds = [
        h["command"]
        for g in doc["hooks"]["SessionStart"]
        for h in g["hooks"]
    ]
    assert "/usr/local/bin/other-tool" in session_cmds
    assert any("aelf-session-start-hook" in c for c in session_cmds)
    # The foreign PreToolUse group survives verbatim, with our Bash
    # group appended after it (#1055).
    pre = doc["hooks"]["PreToolUse"]
    assert pre[0] == foreign["hooks"]["PreToolUse"][0]
    assert any(g.get("matcher") == "Bash" for g in pre[1:])


def test_install_refuses_broken_json_without_force(tmp_path: Path) -> None:
    path = tmp_path / "hooks.json"
    path.write_text('{\n  "broken":', encoding="utf-8")  # observed in the wild
    result = install_codex_hooks(path)
    assert result.error is not None
    assert not result.changed
    # The broken content is preserved for the user to inspect.
    assert path.read_text(encoding="utf-8") == '{\n  "broken":'


def test_install_force_replaces_broken_json(tmp_path: Path) -> None:
    path = tmp_path / "hooks.json"
    path.write_text('{\n  "broken":', encoding="utf-8")
    result = install_codex_hooks(path, force=True)
    assert result.error is None
    assert result.changed
    assert set(_read(path)["hooks"].keys()) == set(desired_codex_hooks().keys())


def test_install_emits_trust_guidance(tmp_path: Path) -> None:
    result = install_codex_hooks(tmp_path / "hooks.json")
    joined = " ".join(result.guidance)
    assert "/hooks" in joined
    # Names the current feature (`hooks`), not the retired `codex_hooks`,
    # and states the correct default (stable / on).
    assert "hooks" in joined
    assert "codex_hooks" not in joined
    assert "on by default" in joined or "enabled by default" in joined


# --- removal ---------------------------------------------------------------


def test_remove_deletes_only_owned_groups(tmp_path: Path) -> None:
    path = tmp_path / "hooks.json"
    install_codex_hooks(path)
    doc = _read(path)
    doc["hooks"]["SessionStart"].insert(
        0, {"hooks": [{"type": "command", "command": "/usr/local/bin/other-tool"}]},
    )
    path.write_text(json.dumps(doc), encoding="utf-8")
    result = remove_codex_hooks(path)
    assert result.changed
    after = _read(path)
    assert set(after["hooks"].keys()) == {"SessionStart"}
    assert after["hooks"]["SessionStart"][0]["hooks"][0]["command"] == (
        "/usr/local/bin/other-tool"
    )


def test_remove_never_touches_broken_json(tmp_path: Path) -> None:
    path = tmp_path / "hooks.json"
    path.write_text('{\n  "broken":', encoding="utf-8")
    result = remove_codex_hooks(path)
    assert result.error is not None
    assert path.read_text(encoding="utf-8") == '{\n  "broken":'


def test_remove_missing_file_is_noop(tmp_path: Path) -> None:
    result = remove_codex_hooks(tmp_path / "hooks.json")
    assert not result.changed
    assert result.error is None


# --- doctor ----------------------------------------------------------------


def test_doctor_missing_codex_dir(tmp_path: Path) -> None:
    report = doctor_codex(tmp_path / "nope")
    assert isinstance(report, CodexDoctorReport)
    assert not report.codex_dir_present
    assert report.warnings


def test_doctor_flags_broken_hooks_json(tmp_path: Path) -> None:
    (tmp_path / "hooks.json").write_text('{\n  "broken":', encoding="utf-8")
    report = doctor_codex(tmp_path)
    assert report.hooks_file_present
    assert not report.hooks_file_valid
    assert report.parse_error


def test_doctor_healthy_install_reports_coverage(tmp_path: Path) -> None:
    install_codex_hooks(tmp_path / "hooks.json")
    (tmp_path / "config.toml").write_text(
        "[features]\nhooks = true\n", encoding="utf-8",
    )
    report = doctor_codex(tmp_path)
    assert report.hooks_file_valid
    assert report.owned_handler_count >= 6
    assert report.missing_events == []
    assert report.feature_flag_on is True
    # Handlers configured but zero approvals -> unapproved warning.
    assert any("unapproved" in w for w in report.warnings)


def test_doctor_feature_on_by_default_when_unmentioned(tmp_path: Path) -> None:
    # Codex 0.145+: `hooks` is stable and on by default, so it is absent
    # from config.toml at its default. Absence must read as ON, and no
    # "feature disabled" warning must fire (regression guard for #1151:
    # the retired `codex_hooks` probe reported this case as off).
    install_codex_hooks(tmp_path / "hooks.json")
    (tmp_path / "config.toml").write_text("model = \"x\"\n", encoding="utf-8")
    report = doctor_codex(tmp_path)
    assert report.feature_flag_on is True
    assert not any("feature is disabled" in w for w in report.warnings)


def test_doctor_warns_only_on_explicit_disable(tmp_path: Path) -> None:
    install_codex_hooks(tmp_path / "hooks.json")
    (tmp_path / "config.toml").write_text(
        "[features]\nhooks = false\n", encoding="utf-8",
    )
    report = doctor_codex(tmp_path)
    assert report.feature_flag_on is False
    assert any("hooks` feature is disabled" in w for w in report.warnings)


def test_doctor_honours_legacy_codex_hooks_key(tmp_path: Path) -> None:
    # An explicit legacy `codex_hooks = true` (Codex 0.11x–0.12x) is still
    # read as enabled, so an upgrader who set it does not get a false off.
    install_codex_hooks(tmp_path / "hooks.json")
    (tmp_path / "config.toml").write_text(
        "[features]\ncodex_hooks = true\n", encoding="utf-8",
    )
    report = doctor_codex(tmp_path)
    assert report.feature_flag_on is True


def test_doctor_counts_approved_state_entries(tmp_path: Path) -> None:
    install_codex_hooks(tmp_path / "hooks.json")
    (tmp_path / "config.toml").write_text(
        '[features]\nhooks = true\n'
        '[hooks.state."k1"]\ntrusted_hash = "abc"\n'
        '[hooks.state."k2"]\ntrusted_hash = "def"\n',
        encoding="utf-8",
    )
    report = doctor_codex(tmp_path)
    assert report.approved_state_count == 2
    # Approvals exist -> the zero-approvals warning must NOT fire.
    assert not any("unapproved" in w for w in report.warnings)
