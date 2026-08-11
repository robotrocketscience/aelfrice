"""uv-tool detection across platforms and configured directories (#1431).

Detection hard-coded `~/.local/share/uv/tools`, so a perfectly normal
Windows uv-tool install classified as `non_uv` and `aelf upgrade-cmd`
recommended `pip uninstall -y aelfrice && uv tool install aelfrice` — advice
that invokes an unrelated pip and leaves lifecycle state confusing.

Every Windows branch here is reached from a POSIX runner by patching
`lifecycle._is_windows`. Patching `os.name` itself is not an option:
`pathlib.Path()` dispatches on it and raises `UnsupportedOperation: cannot
instantiate 'WindowsPath' on your system` the moment a POSIX runner claims to
be Windows. A `windows: bool = os.name == "nt"` default argument would bind
once at definition time and make these tests pass on main too (#1412 review) —
`test_windows_default_is_not_a_bound_default` pins that directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from aelfrice import lifecycle


def _receipt(env_root: Path) -> Path:
    """Create a uv tool environment complete with uv's receipt."""
    env_root.mkdir(parents=True, exist_ok=True)
    receipt = env_root / lifecycle.UV_RECEIPT_FILENAME
    receipt.write_text('[tool]\nrequirements = ["aelfrice"]\n', encoding="utf-8")
    return receipt


# --- the tools directory -------------------------------------------------


def test_uv_tool_dir_honours_the_configured_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`UV_TOOL_DIR` wins outright, on every platform."""
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "custom tools"))
    assert lifecycle._uv_tool_dir() == tmp_path / "custom tools"

    monkeypatch.setattr(lifecycle, "_is_windows", lambda: True)
    assert lifecycle._uv_tool_dir() == tmp_path / "custom tools"


def test_uv_tool_dir_windows_default_is_appdata_uv_data_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The Windows default is %APPDATA%\\uv\\data\\tools.

    uv's storage reference (docs.astral.sh/uv/reference/storage) puts the
    *persistent data directory* at `%APPDATA%\\uv\\data` on Windows — where
    the POSIX one is `$XDG_DATA_HOME/uv` with no `data` component — and
    installs tools into a `tools/` subdirectory of it. `%APPDATA%\\uv\\tools`
    names no directory uv ever writes.
    """
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.setattr(lifecycle, "_is_windows", lambda: True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert (
        lifecycle._uv_tool_dir()
        == tmp_path / "Roaming" / "uv" / "data" / "tools"
    )
    # The asymmetry is the point: the POSIX default has no `data` level,
    # so a shared spelling cannot be right for both.
    monkeypatch.setattr(lifecycle, "_is_windows", lambda: False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert lifecycle._uv_tool_dir() == tmp_path / "xdg" / "uv" / "tools"


def test_uv_tool_dir_posix_default_follows_xdg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """POSIX honours XDG_DATA_HOME, falling back to ~/.local/share."""
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.setattr(lifecycle, "_is_windows", lambda: False)

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert lifecycle._uv_tool_dir() == tmp_path / "xdg" / "uv" / "tools"

    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    assert (
        lifecycle._uv_tool_dir()
        == tmp_path / ".local" / "share" / "uv" / "tools"
    )


def test_windows_default_is_not_a_bound_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The platform must be read per call, not captured at import.

    This is the test that would have caught #1412's inert flag: if the
    branch were selected by a default argument, flipping the probe after
    import would change nothing and both assertions below would return the
    same directory.
    """
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))

    monkeypatch.setattr(lifecycle, "_is_windows", lambda: False)
    posix_dir = lifecycle._uv_tool_dir()
    monkeypatch.setattr(lifecycle, "_is_windows", lambda: True)
    windows_dir = lifecycle._uv_tool_dir()

    assert posix_dir != windows_dir
    assert windows_dir == tmp_path / "Roaming" / "uv" / "data" / "tools"


# --- classification ------------------------------------------------------


def test_windows_uv_tool_install_is_classified_uv_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The headline defect: a default Windows uv layout, end to end.

    On main this returns `non_uv` and a `pip uninstall` migration command.
    """
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.setattr(lifecycle, "_is_windows", lambda: True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    tool_env = tmp_path / "Roaming" / "uv" / "data" / "tools" / "aelfrice"
    _receipt(tool_env)
    # The running process is elsewhere; disk presence alone must suffice
    # for *advice*, which is what upgrade_advice() asks.
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "somewhere" / "else"))
    monkeypatch.setattr(
        sys, "executable", str(tmp_path / "somewhere" / "else" / "python.exe"),
    )

    assert lifecycle._is_uv_tool_install() is True
    advice = lifecycle.upgrade_advice()
    assert advice.context == "uv_tool"
    assert advice.command == "uv tool upgrade aelfrice"
    assert "pip uninstall" not in advice.command


def test_receipt_identifies_the_running_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """uv's receipt at sys.prefix beats any assumption about locations.

    A user whose tools directory is somewhere this code has never heard of
    is still correctly identified, because the running environment says so.
    """
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "not-where-we-look"))
    running = tmp_path / "an" / "unexpected" / "place" / "aelfrice"
    _receipt(running)
    monkeypatch.setattr(sys, "prefix", str(running))
    monkeypatch.setattr(sys, "executable", str(running / "bin" / "python"))

    assert lifecycle._running_from_uv_tool() is True


def test_a_forged_directory_without_a_receipt_is_not_a_uv_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Presence of a directory named `aelfrice` proves nothing.

    Recommending `uv tool upgrade` for a left-over directory hands the user
    a command that cannot succeed.
    """
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "tools"))
    (tmp_path / "tools" / "aelfrice").mkdir(parents=True)
    elsewhere = tmp_path / "venv"
    monkeypatch.setattr(sys, "prefix", str(elsewhere))
    monkeypatch.setattr(sys, "executable", str(elsewhere / "bin" / "python"))

    assert lifecycle._is_uv_tool_install() is False
    assert lifecycle.upgrade_advice().context == "non_uv"


def test_uv_run_from_a_worktree_is_not_the_tool_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """#1044 regression, re-pinned against the configurable root."""
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "tools"))
    _receipt(tmp_path / "tools" / "aelfrice")
    worktree_venv = tmp_path / "projects" / "aelfrice" / ".venv"
    monkeypatch.setattr(sys, "prefix", str(worktree_venv))
    monkeypatch.setattr(
        sys, "executable", str(worktree_venv / "bin" / "python"),
    )

    assert lifecycle._is_uv_tool_install() is True   # it exists on the box
    assert lifecycle._running_from_uv_tool() is False  # but we are not it


def test_a_sibling_directory_is_not_under_the_tools_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """`.../uv/toolshed` must not satisfy a test against `.../uv/tools`.

    The old string-prefix compare accepted it; ancestry does not.
    """
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "uv" / "tools"))
    sibling = tmp_path / "uv" / "toolshed" / "aelfrice"
    sibling.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(sibling))
    monkeypatch.setattr(sys, "executable", str(sibling / "bin" / "python"))

    assert lifecycle._running_from_uv_tool() is False


# --- PATH scanning -------------------------------------------------------


def test_path_scan_finds_the_windows_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """On Windows the launcher is `aelf` + a PATHEXT suffix.

    There is no executable bit to test, so requiring one — as the POSIX-only
    scan did — finds nothing and the reachable-install inventory disagrees
    with the classifier.
    """
    bin_dir = tmp_path / "Scripts"
    bin_dir.mkdir()
    # Named to match what the scan joins. PATHEXT entries are upper-case on
    # a real Windows box, so the probe is `aelf.EXE` — a fixture spelled
    # `aelf.exe` matches on a case-insensitive macOS volume and does NOT on
    # a case-sensitive Linux runner, which is how a green local run turns
    # red on CI. Keep the case aligned rather than relying on the volume.
    launcher = bin_dir / "aelf.EXE"
    launcher.write_bytes(b"MZ")  # not executable; on Windows nothing is

    monkeypatch.setattr(lifecycle, "_is_windows", lambda: True)
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    monkeypatch.setenv("PATH", str(bin_dir))

    found = lifecycle._which_all_aelf()
    assert len(found) == 1
    assert found[0].parent == bin_dir
    assert found[0].name.lower() == "aelf.exe"


def test_path_scan_honours_a_lowercase_pathext(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """PATHEXT is read verbatim, so a lower-case entry resolves too."""
    bin_dir = tmp_path / "Scripts"
    bin_dir.mkdir()
    (bin_dir / "aelf.cmd").write_text("@echo off\n", encoding="utf-8")

    monkeypatch.setattr(lifecycle, "_is_windows", lambda: True)
    monkeypatch.setenv("PATHEXT", ".exe;.cmd")
    monkeypatch.setenv("PATH", str(bin_dir))

    assert [p.name for p in lifecycle._which_all_aelf()] == ["aelf.cmd"]


def test_path_scan_still_requires_the_executable_bit_on_posix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The POSIX contract is unchanged: a non-executable file is not a
    launcher, and `aelf.exe` is not a POSIX console script name."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "aelf").write_text("#!/bin/sh\n", encoding="utf-8")
    (bin_dir / "aelf.exe").write_bytes(b"MZ")

    monkeypatch.setattr(lifecycle, "_is_windows", lambda: False)
    monkeypatch.setenv("PATH", str(bin_dir))
    assert lifecycle._which_all_aelf() == []

    (bin_dir / "aelf").chmod(0o755)
    assert [p.name for p in lifecycle._which_all_aelf()] == ["aelf"]


# --- the classifier and the inventory agree ------------------------------


@pytest.mark.parametrize("with_receipt", [True, False])
def test_classifier_and_inventory_agree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, with_receipt: bool,
) -> None:
    """One state, one answer — in BOTH states.

    Nothing pinned this. `_is_uv_tool_install()` tested for uv's receipt
    while `detect_reachable_installs()` tested `.exists()`, so a bare
    `<tools>/aelfrice/` was simultaneously `non_uv` to `upgrade_advice()`
    and a `uv_tool` site in the multi-install warning — the inventory
    naming an install the upgrade path denied existed. Sharing the
    directory resolver is not what makes them agree; sharing the
    predicate is.
    """
    tool_env = tmp_path / "tools" / "aelfrice"
    tool_env.mkdir(parents=True)
    if with_receipt:
        _receipt(tool_env)
    monkeypatch.setenv("UV_TOOL_DIR", str(tmp_path / "tools"))
    monkeypatch.setenv("PATH", str(tmp_path / "nothing-here"))
    # The running process is a plain venv elsewhere, so `_running_from_
    # uv_tool` cannot supply the answer by itself.
    elsewhere = tmp_path / "venv"
    monkeypatch.setattr(sys, "prefix", str(elsewhere))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base_python"))
    monkeypatch.setattr(sys, "executable", str(elsewhere / "bin" / "python"))

    classified = lifecycle._is_uv_tool_install()
    inventoried = any(
        s.kind == "uv_tool" for s in lifecycle.detect_reachable_installs()
    )

    assert classified is with_receipt
    assert inventoried is with_receipt
    assert classified == inventoried
    assert (lifecycle.upgrade_advice().context == "uv_tool") == inventoried


# --- the auto-install gate stays shut on Windows -------------------------


def _windows_uv_tool_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Make this process look like a default-layout Windows uv-tool install."""
    monkeypatch.delenv("UV_TOOL_DIR", raising=False)
    monkeypatch.setattr(lifecycle, "_is_windows", lambda: True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    tool_env = (
        tmp_path / "Roaming" / "uv" / "data" / "tools" / "aelfrice"
    )
    _receipt(tool_env)
    monkeypatch.setattr(sys, "prefix", str(tool_env))
    monkeypatch.setattr(
        sys, "executable", str(tool_env / "Scripts" / "python.exe"),
    )


def test_auto_install_gate_stays_shut_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Operator ruling on #1431: fix the classifier, keep the gate shut.

    The pairing is the whole point. `lifecycle._running_from_uv_tool()`
    returning True is the classifier working — that is the fix. The gate
    reading False on the *same* state is the ruling: correcting the
    classifier would otherwise let `aelf` start rewriting
    `~/.claude/settings.json` on a platform where it never has, because
    the hook commands it writes resolve through a `bin` literal that is a
    no-op on Windows.

    A test that only asserted the gate is False would pass on a broken
    classifier too, which is why both halves are asserted together.
    """
    from aelfrice import auto_install

    _windows_uv_tool_process(monkeypatch, tmp_path)
    monkeypatch.setattr(auto_install, "_is_windows", lambda: True)

    assert lifecycle._running_from_uv_tool() is True
    assert auto_install.is_running_from_uv_tool_install() is False


def test_auto_install_gate_is_open_for_the_same_state_on_posix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The platform short-circuit, and only it, is what shuts the gate.

    Identical process state; flip only the platform probe. Without this
    the previous test is satisfied by a gate wired shut for any reason.
    """
    from aelfrice import auto_install

    _windows_uv_tool_process(monkeypatch, tmp_path)
    monkeypatch.setattr(auto_install, "_is_windows", lambda: False)

    assert lifecycle._running_from_uv_tool() is True
    assert auto_install.is_running_from_uv_tool_install() is True


def test_auto_install_at_cli_entry_does_not_merge_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The gate is the only thing between that state and a settings write.

    `auto_install_at_cli_entry` returns early on exactly three conditions:
    the env opt-out, the host opt-out, and this gate. With the first two
    unset, a merge attempt here is a real `~/.claude/settings.json`
    rewrite on Windows.
    """
    from aelfrice import auto_install

    _windows_uv_tool_process(monkeypatch, tmp_path)
    monkeypatch.setattr(auto_install, "_is_windows", lambda: True)
    monkeypatch.delenv("AELFRICE_NO_AUTO_INSTALL", raising=False)
    monkeypatch.setattr(auto_install, "read_host_opt_outs", lambda _p: set())

    called: list[str] = []

    def _boom(*, installed_version: str) -> None:
        called.append(installed_version)
        raise AssertionError("auto-install merged on Windows")

    monkeypatch.setattr(auto_install, "maybe_install_manifest", _boom)
    auto_install.auto_install_at_cli_entry("9.9.9")
    assert called == []
