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


def test_format_update_banner_text() -> None:
    """Banner text matches the canonical shortened format."""
    banner = lifecycle.format_update_banner("1.5.0")
    assert banner == "⬆ /aelf:upgrade to v1.5.0"


def test_format_update_banner_contains_slash_command() -> None:
    assert "/aelf:upgrade" in lifecycle.format_update_banner("2.0.0")


def test_statusline_snippet_uses_banner_helper() -> None:
    """statusline.format_snippet embeds the output of format_update_banner."""
    status = lifecycle.UpdateStatus(
        update_available=True,
        installed="1.0.0",
        latest="1.5.0",
        checked=0.0,
        sha256=None,
    )
    snippet = statusline.format_snippet(status, env={})
    expected_body = lifecycle.format_update_banner("1.5.0")
    assert expected_body in snippet


def test_banner_shared_substring_in_statusline_and_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both call sites derive text from format_update_banner.

    This test verifies they share the '/aelf:upgrade to v' substring,
    ensuring the two surfaces can never drift from each other.
    """
    import io

    status = lifecycle.UpdateStatus(
        update_available=True,
        installed="1.0.0",
        latest="1.5.0",
        checked=0.0,
        sha256=None,
    )
    # Statusline path.
    snippet = statusline.format_snippet(status, env={})
    assert "/aelf:upgrade to v" in snippet

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
    assert "/aelf:upgrade to v" in stderr_out


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
