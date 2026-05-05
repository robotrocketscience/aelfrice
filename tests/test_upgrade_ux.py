"""Tests for issue #242: upgrade UX — banner format, version source-of-truth,
install-method detection.

All tests are hermetic: filesystem, sys.prefix, and importlib.metadata are
monkeypatched so no real install state bleeds in.
"""
from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

from aelfrice import lifecycle, statusline


# ---------------------------------------------------------------------------
# installed_version — importlib.metadata path
# ---------------------------------------------------------------------------


def test_installed_version_reads_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """installed_version() returns the value from importlib.metadata."""
    import importlib.metadata as _meta

    monkeypatch.setattr(_meta, "version", lambda pkg: "9.8.7")
    assert lifecycle.installed_version() == "9.8.7"


def test_installed_version_fallback_on_package_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """installed_version() returns '0.0.0' when the package isn't found."""
    import importlib.metadata as _meta

    def _raise(pkg: str) -> str:
        raise PackageNotFoundError(pkg)

    monkeypatch.setattr(_meta, "version", _raise)
    assert lifecycle.installed_version() == "0.0.0"


# ---------------------------------------------------------------------------
# format_update_banner — single source of truth
# ---------------------------------------------------------------------------


def test_format_update_banner_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Banner embeds the install-method-aware command from upgrade_advice."""
    monkeypatch.setattr(
        lifecycle,
        "upgrade_advice",
        lambda: lifecycle.UpgradeAdvice(
            command="uv tool upgrade aelfrice", context="uv_tool"
        ),
    )
    banner = lifecycle.format_update_banner("1.5.0")
    assert banner == "⬆ aelfrice 1.5.0 — run: uv tool upgrade aelfrice"


def test_format_update_banner_explicit_command_kwarg() -> None:
    """Caller-supplied `command` overrides upgrade_advice() lookup."""
    banner = lifecycle.format_update_banner(
        "2.0.0", command="pipx upgrade aelfrice"
    )
    assert banner == "⬆ aelfrice 2.0.0 — run: pipx upgrade aelfrice"


def test_format_update_banner_no_slash_command_advisory() -> None:
    """Banner must not phrase a slash command as the upgrade action.

    Regression for #427: the previous shape `'⬆ /aelf:upgrade to v…'`
    read as an imperative pointing at a slash command that is itself
    advisory, leading users to believe the slash command performed the
    upgrade. The banner must surface the actual shell line instead.
    """
    banner = lifecycle.format_update_banner(
        "1.5.0", command="uv tool upgrade aelfrice"
    )
    assert "/aelf:upgrade" not in banner


def test_statusline_snippet_uses_banner_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """statusline.format_snippet embeds the output of format_update_banner."""
    monkeypatch.setattr(
        lifecycle,
        "upgrade_advice",
        lambda: lifecycle.UpgradeAdvice(
            command="uv tool upgrade aelfrice", context="uv_tool"
        ),
    )
    status = lifecycle.UpdateStatus(
        update_available=True,
        installed="1.0.0",
        latest="1.5.0",
        checked=0.0,
        sha256=None,
    )
    # Pass installed="" so the snippet is not suppressed by the running
    # package version (which is >= status.latest in this worktree).
    snippet = statusline.format_snippet(status, env={}, installed="")
    expected_body = lifecycle.format_update_banner("1.5.0")
    assert expected_body in snippet


def test_banner_shared_substring_in_statusline_and_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both call sites derive text from format_update_banner.

    This test verifies they share the 'aelfrice <ver> — run:' substring,
    ensuring the two surfaces can never drift from each other.
    """
    import io

    monkeypatch.setattr(
        lifecycle,
        "upgrade_advice",
        lambda: lifecycle.UpgradeAdvice(
            command="uv tool upgrade aelfrice", context="uv_tool"
        ),
    )
    status = lifecycle.UpdateStatus(
        update_available=True,
        installed="1.0.0",
        latest="1.5.0",
        checked=0.0,
        sha256=None,
    )
    # Statusline path. Pass installed="" so the running package version
    # does not auto-suppress the snippet.
    snippet = statusline.format_snippet(status, env={}, installed="")
    assert "aelfrice 1.5.0 — run:" in snippet

    # CLI stderr banner path — monkeypatch the cache reader.
    import aelfrice.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_read_update_cache", lambda: status)
    monkeypatch.setattr(
        cli_mod, "_update_check_disabled", lambda: False
    )

    buf = io.StringIO()
    # _maybe_emit_update_banner writes to sys.stderr; capture it.
    monkeypatch.setattr(sys, "stderr", buf)
    cli_mod._maybe_emit_update_banner("search")
    stderr_out = buf.getvalue()
    assert "aelfrice 1.5.0 — run:" in stderr_out


# ---------------------------------------------------------------------------
# install-method detection — each branch
# ---------------------------------------------------------------------------


def test_is_uv_tool_install_via_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """uv-tool install detected when package dir exists under uv tools root."""
    fake_uv_tools = tmp_path / ".local" / "share" / "uv" / "tools"
    (fake_uv_tools / "aelfrice").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert lifecycle._is_uv_tool_install() is True


def test_is_uv_tool_install_via_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """uv-tool install detected when sys.prefix is under the uv tools root."""
    uv_tools_root = tmp_path / ".local" / "share" / "uv" / "tools"
    uv_tools_root.mkdir(parents=True)
    fake_prefix = str(uv_tools_root / "aelfrice" / "env")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(sys, "prefix", fake_prefix)
    assert lifecycle._is_uv_tool_install() is True


def test_is_uv_tool_install_false_when_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No uv-tool install detected when neither directory nor prefix matches."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    # sys.prefix points into a plain venv that has nothing to do with uv tools.
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "my_venv"))
    monkeypatch.setattr(
        sys, "base_prefix", str(tmp_path / "base_python")
    )
    assert lifecycle._is_uv_tool_install() is False


def test_is_pipx_install_via_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """pipx install detected when sys.prefix contains /pipx/venvs/."""
    monkeypatch.setattr(
        sys, "prefix", "/home/user/.local/pipx/venvs/aelfrice"
    )
    assert lifecycle._is_pipx_install() is True


def test_is_pipx_install_via_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """pipx install detected when the pipx venv directory exists."""
    fake_home = tmp_path
    pipx_venv = fake_home / ".local" / "pipx" / "venvs" / "aelfrice"
    pipx_venv.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "unrelated"))
    assert lifecycle._is_pipx_install() is True


def test_is_pipx_install_false_when_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "my_venv"))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert lifecycle._is_pipx_install() is False


def test_is_venv_true_when_prefix_differs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "prefix", "/some/venv")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    assert lifecycle._is_venv() is True


def test_is_venv_false_when_prefix_equals_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "prefix", "/usr")
    monkeypatch.setattr(sys, "base_prefix", "/usr")
    assert lifecycle._is_venv() is False


# ---------------------------------------------------------------------------
# upgrade_advice — full routing table
# ---------------------------------------------------------------------------


def _patch_detection(
    monkeypatch: pytest.MonkeyPatch,
    *,
    uv_tool: bool = False,
    pipx: bool = False,
    venv: bool = False,
) -> None:
    monkeypatch.setattr(lifecycle, "_is_uv_tool_install", lambda: uv_tool)
    monkeypatch.setattr(lifecycle, "_is_pipx_install", lambda: pipx)
    monkeypatch.setattr(lifecycle, "_is_venv", lambda: venv)


@pytest.mark.parametrize(
    "uv_tool,pipx,venv,expected_context,expected_cmd_fragment",
    [
        (True, False, False, "uv_tool", "uv tool upgrade aelfrice"),
        (False, True, False, "pipx", "pipx upgrade aelfrice"),
        (False, False, True, "venv", "pip install --upgrade aelfrice"),
        (False, False, False, "system", "pip install --user --upgrade aelfrice"),
        # uv_tool wins even if venv is also true (order check).
        (True, False, True, "uv_tool", "uv tool upgrade aelfrice"),
        # pipx wins over plain venv.
        (False, True, True, "pipx", "pipx upgrade aelfrice"),
    ],
)
def test_upgrade_advice_routing(
    monkeypatch: pytest.MonkeyPatch,
    uv_tool: bool,
    pipx: bool,
    venv: bool,
    expected_context: str,
    expected_cmd_fragment: str,
) -> None:
    _patch_detection(monkeypatch, uv_tool=uv_tool, pipx=pipx, venv=venv)
    advice = lifecycle.upgrade_advice()
    assert advice.context == expected_context
    assert advice.command == expected_cmd_fragment


# ---------------------------------------------------------------------------
# detect_reachable_installs — multi-install warning (issue #345)
# ---------------------------------------------------------------------------


def _make_aelf_exe(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)


def test_reachable_installs_empty_when_nothing_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv("PATH", str(tmp_path / "empty_bin"))
    assert lifecycle.detect_reachable_installs() == []


def test_reachable_installs_single_uv_tool_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    uv_root = tmp_path / ".local" / "share" / "uv" / "tools" / "aelfrice"
    uv_root.mkdir(parents=True)
    monkeypatch.setenv("PATH", str(tmp_path / "empty_bin"))
    sites = lifecycle.detect_reachable_installs()
    assert [(s.kind, s.on_path) for s in sites] == [("uv_tool", False)]


def test_reachable_installs_uv_tool_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    uv_root = tmp_path / ".local" / "share" / "uv" / "tools" / "aelfrice"
    uv_bin_dir = uv_root / "bin"
    _make_aelf_exe(uv_bin_dir / "aelf")
    monkeypatch.setenv("PATH", str(uv_bin_dir))
    sites = lifecycle.detect_reachable_installs()
    assert [(s.kind, s.on_path) for s in sites] == [("uv_tool", True)]


def test_reachable_installs_dual_uv_tool_plus_user_local_bin(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Repro from issue #318: uv-tool install + stale ~/.local/bin/aelf on PATH.

    Both installs detected; the user_local_bin one is marked on_path,
    the uv_tool one is not.
    """
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    uv_root = tmp_path / ".local" / "share" / "uv" / "tools" / "aelfrice"
    uv_root.mkdir(parents=True)
    user_bin = tmp_path / ".local" / "bin"
    _make_aelf_exe(user_bin / "aelf")
    monkeypatch.setenv("PATH", str(user_bin))

    sites = lifecycle.detect_reachable_installs()
    by_kind = {s.kind: s for s in sites}
    assert set(by_kind) == {"uv_tool", "user_local_bin"}
    assert by_kind["uv_tool"].on_path is False
    assert by_kind["user_local_bin"].on_path is True
    assert by_kind["user_local_bin"].path == user_bin / "aelf"


def test_reachable_installs_pipx_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    pipx_root = tmp_path / ".local" / "pipx" / "venvs" / "aelfrice"
    pipx_root.mkdir(parents=True)
    monkeypatch.setenv("PATH", str(tmp_path / "empty_bin"))
    sites = lifecycle.detect_reachable_installs()
    assert [(s.kind, s.on_path) for s in sites] == [("pipx", False)]


def test_format_multi_install_warning_silent_when_single(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aelfrice import cli

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    uv_root = tmp_path / ".local" / "share" / "uv" / "tools" / "aelfrice"
    uv_root.mkdir(parents=True)
    monkeypatch.setenv("PATH", str(tmp_path / "empty_bin"))
    sites = lifecycle.detect_reachable_installs()
    assert cli._format_multi_install_warning(sites, "uv_tool") == []


def test_format_multi_install_warning_renders_when_multiple(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aelfrice import cli

    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    uv_root = tmp_path / ".local" / "share" / "uv" / "tools" / "aelfrice"
    uv_root.mkdir(parents=True)
    user_bin = tmp_path / ".local" / "bin"
    _make_aelf_exe(user_bin / "aelf")
    monkeypatch.setenv("PATH", str(user_bin))
    sites = lifecycle.detect_reachable_installs()
    lines = cli._format_multi_install_warning(sites, "uv_tool")
    body = "\n".join(lines)
    assert "warning: multiple aelfrice installs detected" in body
    assert "uv tool:" in body
    assert "user-local:" in body
    assert "(on PATH)" in body
    assert "(uv_tool)" in body


def test_reachable_installs_path_aelf_under_uv_root_not_double_counted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An `aelf` exe living under the uv-tool root must not register as a
    separate user_local_bin entry."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    uv_root = tmp_path / ".local" / "share" / "uv" / "tools" / "aelfrice"
    uv_bin_dir = uv_root / "bin"
    _make_aelf_exe(uv_bin_dir / "aelf")
    monkeypatch.setenv("PATH", str(uv_bin_dir))
    sites = lifecycle.detect_reachable_installs()
    assert len(sites) == 1
    assert sites[0].kind == "uv_tool"
