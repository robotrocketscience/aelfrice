"""`aelf setup` <-> auto_install opt-out + stamp sync (#623)."""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice import auto_install, cli


def _patch_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    """Redirect STAMP_PATH and OPT_OUT_PATH at the module attribute level.

    `cli._sync_setup_opt_outs_and_stamp` reads
    `_auto_install.STAMP_PATH` and `_auto_install.OPT_OUT_PATH` at call
    time, so patching the auto_install module's globals is sufficient.
    """
    stamp = tmp_path / "stamp"
    opt_out = tmp_path / "opt-out.json"
    monkeypatch.setattr(auto_install, "STAMP_PATH", stamp)
    monkeypatch.setattr(auto_install, "OPT_OUT_PATH", opt_out)
    return stamp, opt_out


def test_setup_writes_stamp_at_installed_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    stamp, _ = _patch_paths(monkeypatch, tmp_path)
    settings = tmp_path / "settings.json"
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    code = cli.main([
        "setup", "--settings", str(settings), "--scope", "project",
        "--no-statusline",
    ])
    assert code == 0
    assert auto_install.read_stamp(stamp) == cli._AELFRICE_VERSION


def test_setup_no_transcript_ingest_adds_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _, opt_out = _patch_paths(monkeypatch, tmp_path)
    settings = tmp_path / "settings.json"
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    code = cli.main([
        "setup", "--settings", str(settings), "--scope", "project",
        "--no-statusline", "--no-transcript-ingest",
    ])
    assert code == 0
    assert "transcript_ingest" in auto_install.read_opt_outs(opt_out)


def test_setup_rescinds_opt_out_when_user_re_enables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """User opted out, ran a bare upgrade, then re-ran `aelf setup` without
    --no-X. The opt-out must be dropped so the hook gets installed."""
    _, opt_out = _patch_paths(monkeypatch, tmp_path)
    settings = tmp_path / "settings.json"
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    auto_install.add_opt_out("transcript_ingest", opt_out)
    assert "transcript_ingest" in auto_install.read_opt_outs(opt_out)
    cli.main([
        "setup", "--settings", str(settings), "--scope", "project",
        "--no-statusline",
    ])
    assert "transcript_ingest" not in auto_install.read_opt_outs(opt_out)


def test_setup_no_stop_hook_uses_correct_manifest_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """--no-stop-hook maps to manifest name 'stop_lock_prompt', not 'stop_hook'."""
    _, opt_out = _patch_paths(monkeypatch, tmp_path)
    settings = tmp_path / "settings.json"
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    cli.main([
        "setup", "--settings", str(settings), "--scope", "project",
        "--no-statusline", "--no-stop-hook",
    ])
    opt_outs = auto_install.read_opt_outs(opt_out)
    assert "stop_lock_prompt" in opt_outs
    assert "stop_hook" not in opt_outs


def test_setup_no_search_tool_adds_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`aelf setup --no-search-tool` persists the opt-out (#738)."""
    _, opt_out = _patch_paths(monkeypatch, tmp_path)
    settings = tmp_path / "settings.json"
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    cli.main([
        "setup", "--settings", str(settings), "--scope", "project",
        "--no-statusline", "--no-search-tool",
    ])
    assert "search_tool" in auto_install.read_opt_outs(opt_out)


def test_setup_no_search_tool_bash_adds_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`aelf setup --no-search-tool-bash` persists the opt-out (#738)."""
    _, opt_out = _patch_paths(monkeypatch, tmp_path)
    settings = tmp_path / "settings.json"
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    cli.main([
        "setup", "--settings", str(settings), "--scope", "project",
        "--no-statusline", "--no-search-tool-bash",
    ])
    assert "search_tool_bash" in auto_install.read_opt_outs(opt_out)


def test_setup_bare_rescinds_search_tool_opt_outs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Bare `aelf setup` after a prior --no-search-tool drops the opt-out (#738)."""
    _, opt_out = _patch_paths(monkeypatch, tmp_path)
    settings = tmp_path / "settings.json"
    monkeypatch.setenv("AELF_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("HOME", str(tmp_path))
    auto_install.add_opt_out("search_tool", opt_out)
    auto_install.add_opt_out("search_tool_bash", opt_out)
    cli.main([
        "setup", "--settings", str(settings), "--scope", "project",
        "--no-statusline",
    ])
    opt_outs = auto_install.read_opt_outs(opt_out)
    assert "search_tool" not in opt_outs
    assert "search_tool_bash" not in opt_outs
