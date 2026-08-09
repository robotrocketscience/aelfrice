"""#1412: the shared console-launcher primitive.

Nobody on the team has a Windows host, so — following the pattern
`test_windows_portability_1329.py` established — every arm here drives the
Windows branch explicitly from POSIX CI rather than skipping. The `windows=`
keyword is the seam that makes that possible; `test_the_platform_flag_is_not
_bound_at_definition_time` is the arm that proves the seam is real, because
the natural implementation of it silently is not.
"""
from __future__ import annotations

import os
import sysconfig
from pathlib import Path

import pytest

from aelfrice import launcher

WIN_PATH = r"C:\Users\dev\.venv\Scripts\aelf-hook.EXE"

# `shutil.which` resolves a bare name only against PATHEXT on win32, so a
# fixture launcher must carry a real suffix there and must not on POSIX.
LAUNCHER_NAME = "aelf-hook.exe" if os.name == "nt" else "aelf-hook"


def _install_launcher(directory: Path, name: str = LAUNCHER_NAME) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    return target


def test_the_platform_flag_is_not_bound_at_definition_time() -> None:
    """`os.name` must be read per call, not captured at import.

    The obvious signature — ``def launcher_key(tok, *, windows: bool =
    os.name == "nt")`` — evaluates its default once, when the module is
    imported. Under it this test passes trivially on Linux both before and
    after the real fix, because every call takes the POSIX branch. Asserting
    the *monkeypatched* value is what distinguishes a live seam from a dead
    one.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "name", "nt")
        assert launcher.launcher_key(WIN_PATH) == "aelf-hook"
        assert launcher.command_launcher_key(WIN_PATH) == "aelf-hook"
    # ...and back to the host's real platform once the patch is dropped.
    assert launcher.launcher_key(WIN_PATH, windows=False) == WIN_PATH


class TestWindowsNormalisation:
    @pytest.mark.parametrize("token", [
        r"C:\Users\dev\.venv\Scripts\aelf-hook.EXE",
        r"C:\Users\dev\.venv\Scripts\aelf-hook.exe",
        r"C:\Tools\Aelf-Hook.CMD",
        r"C:\Tools\AELF-HOOK.bat",
        r"C:\Tools\aelf-hook.com",
        "aelf-hook",
        "aelf-hook.exe",
        "C:/forward/slashes/aelf-hook.exe",
    ])
    def test_every_windows_launcher_spelling_maps_to_one_key(
        self, token: str,
    ) -> None:
        assert launcher.launcher_key(token, windows=True) == "aelf-hook"

    def test_a_non_launcher_suffix_is_significant(self) -> None:
        """Only PATHEXT-style launcher suffixes are stripped."""
        assert launcher.launcher_key(
            r"C:\Tools\aelf-hook.py", windows=True,
        ) == "aelf-hook.py"

    def test_posix_normalisation_is_unchanged(self) -> None:
        """Case and suffix stay significant on POSIX.

        Folding unconditionally would widen ownership on a case-sensitive
        filesystem, and ownership drives `remove_codex_hooks` and
        `prune_broken_aelf_hooks` — both of which delete.
        """
        assert launcher.launcher_key(
            "/usr/local/bin/Aelf-Hook.EXE", windows=False,
        ) == "Aelf-Hook.EXE"
        assert launcher.launcher_key(
            "/usr/local/bin/aelf-hook", windows=False,
        ) == "aelf-hook"


class TestTokenising:
    def test_windows_tokenising_preserves_backslashes(self) -> None:
        """`shlex.split`'s POSIX mode eats them; that is the prune bug.

        ``shlex.split(r"C:\\Scripts\\aelf-hook.exe")`` returns
        ``["C:Scriptsaelf-hook.exe"]``. `doctor._inspect_command` then cannot
        find the program, calls it broken, and `prune_broken_aelf_hooks` —
        which `aelf setup` runs unconditionally — deletes a working install.
        """
        import shlex
        assert shlex.split(WIN_PATH) == ["C:Usersdev.venvScriptsaelf-hook.EXE"]
        assert launcher.command_tokens(WIN_PATH, windows=True) == [WIN_PATH]

    def test_a_quoted_windows_path_with_spaces_is_one_token(self) -> None:
        cmd = r'"C:\Path With Spaces\aelf-hook.exe" --flag'
        assert launcher.command_tokens(cmd, windows=True) == [
            r"C:\Path With Spaces\aelf-hook.exe", "--flag",
        ]
        assert launcher.command_launcher_key(cmd, windows=True) == "aelf-hook"

    def test_posix_tokenising_is_untouched(self) -> None:
        assert launcher.command_tokens(
            "bash /opt/x.sh 2>/dev/null", windows=False,
        ) == ["bash", "/opt/x.sh", "2>/dev/null"]

    def test_unbalanced_quoting_still_yields_a_program(self) -> None:
        assert launcher.command_launcher_key(
            r'"C:\Tools\aelf-hook.exe --flag', windows=True,
        ) == "aelf-hook"

    @pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
    def test_blank_commands_yield_no_program(self, blank: str) -> None:
        assert launcher.program_token(blank, windows=True) == ""
        assert launcher.command_launcher_key(blank, windows=False) == ""


class TestResolution:
    def test_scripts_dir_comes_from_sysconfig_not_a_hardcoded_leaf(
        self,
    ) -> None:
        """The old code was ``Path(sys.prefix) / "bin"``.

        On Windows that names a directory that does not exist, so the venv
        branch of `_resolve_script` was dead and project scope silently fell
        through to a bare `PATH` search.
        """
        assert launcher.scripts_dir() == Path(sysconfig.get_path("scripts"))

    def test_which_in_takes_an_explicit_directory(self, tmp_path: Path) -> None:
        target = _install_launcher(tmp_path)
        assert launcher.which_in(tmp_path, "aelf-hook") == target
        assert launcher.which_in(tmp_path, "aelf-missing") is None

    def test_which_on_path_never_passes_path_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The win32 current-directory search is a hijack vector.

        CPython inserts `os.curdir` into the search path **only when `path is
        None`**. A bare `shutil.which("aelf-hook")` on Windows can therefore
        resolve an `aelf-hook.exe` in whatever directory `aelf setup` was run
        from and pin it into settings.json. Asserting the kwarg is the only
        way to catch a regression to the bare call from POSIX CI.
        """
        seen: list[object] = []

        def fake_which(name: str, mode: int = 1, path: object = None) -> None:
            seen.append(path)
            return None

        monkeypatch.setattr(launcher.shutil, "which", fake_which)
        monkeypatch.setenv("PATH", "/sentinel/bin")
        assert launcher.which_on_path("aelf-hook") is None
        assert seen == ["/sentinel/bin"]

    def test_setup_delegates_the_scripts_dir_rather_than_re_deriving_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """Asserting equality here would prove nothing on POSIX.

        `sysconfig.get_path("scripts")` and the old ``Path(sys.prefix) /
        "bin"`` are the *same directory* on this host, so
        ``assert setup._venv_bin_dir() == launcher.scripts_dir()`` passes on
        a tree that still hardcodes the POSIX leaf. Only the delegation is
        observable from here: patch the primitive and require the caller to
        follow it. The behavioural difference itself is Windows-only and is
        covered by the `windows-smoke` job.
        """
        from aelfrice import setup

        monkeypatch.setattr(launcher, "scripts_dir", lambda: tmp_path)
        assert setup._venv_bin_dir() == tmp_path

    def test_owned_keys_folds_only_under_windows(self) -> None:
        names = frozenset({"aelf-hook", "aelf-stop-hook"})
        assert launcher.owned_keys(names, windows=True) == names
        assert launcher.owned_keys(names, windows=False) == names
        # A launcher spelling resolves into the set only on Windows.
        assert launcher.launcher_key(
            "AELF-HOOK.EXE", windows=True,
        ) in launcher.owned_keys(names, windows=True)
        assert launcher.launcher_key(
            "AELF-HOOK.EXE", windows=False,
        ) not in launcher.owned_keys(names, windows=False)


class TestScriptResolution:
    """`_resolve_script`'s two candidates, and which one wins.

    AC: "a real installed `aelf-hook.EXE` under `Scripts` is preferred over
    an unrelated global shim" for project scope.
    """

    def test_project_scope_prefers_the_interpreter_scripts_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from aelfrice import setup

        scripts = tmp_path / "env" / "Scripts"
        elsewhere = tmp_path / "global"
        local = _install_launcher(scripts)
        _install_launcher(elsewhere)

        monkeypatch.setattr(launcher, "scripts_dir", lambda: scripts)
        monkeypatch.setenv("PATH", str(elsewhere))
        chosen = Path(setup._resolve_script("aelf-hook", "project"))

        # Compared by directory and key, not by string. On Windows
        # `shutil.which` returns the name joined with a PATHEXT entry, and
        # those are upper-case, so a fixture written as `aelf-hook.exe` comes
        # back as `aelf-hook.EXE` and an equality assert fails on case alone.
        # The property under test is *which directory won*, not its spelling.
        assert chosen.parent == scripts
        assert launcher.launcher_key(str(chosen)) == launcher.launcher_key(
            str(local),
        )

    def test_the_scripts_probe_delegates_to_the_pathext_aware_resolver(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The old probe was ``directory / name`` — no extension, so on
        Windows ``aelf-hook`` never found ``aelf-hook.exe``.

        PATHEXT only exists on win32, so POSIX CI cannot observe the
        difference in behaviour; what it can observe is that the probe now
        routes through the resolver that implements it. Deleting that edge
        is the regression this catches.
        """
        seen: list[tuple[Path, str]] = []
        sentinel = tmp_path / "aelf-hook.exe"

        def fake_which_in(directory: Path, name: str) -> Path:
            seen.append((directory, name))
            return sentinel

        monkeypatch.setattr(launcher, "which_in", fake_which_in)
        assert setup_executable(tmp_path, "aelf-hook") == sentinel
        assert seen == [(tmp_path, "aelf-hook")]

    def test_resolve_script_uses_the_curdir_safe_path_lookup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The edge, not the ends.

        `launcher.which_on_path` is covered on its own, and
        `_resolve_script` is covered on its own — but a straight revert to
        ``shutil.which(script_name)`` inside `_resolve_script` passed every
        other arm in this file. Only asserting the connecting call catches
        it, and on POSIX that call is the *whole* difference: the curdir
        insertion it avoids exists only on win32.
        """
        from aelfrice import setup

        shim = tmp_path / "aelf-hook"
        calls: list[str] = []

        def fake_which_on_path(name: str) -> Path:
            calls.append(name)
            return shim

        monkeypatch.setattr(launcher, "which_on_path", fake_which_on_path)
        monkeypatch.setattr(launcher, "which_in", lambda _d, _n: None)
        assert setup._resolve_script("aelf-hook", "user") == str(shim)
        assert calls == ["aelf-hook"]

    def test_a_present_launcher_resolves_and_a_missing_one_does_not(
        self, tmp_path: Path,
    ) -> None:
        installed = _install_launcher(tmp_path / "Scripts")
        assert setup_executable(tmp_path / "Scripts", "aelf-hook") == installed
        assert setup_executable(tmp_path / "Scripts", "aelf-nonesuch") is None

    @pytest.mark.skipif(
        os.name == "nt",
        reason="the executable bit is a POSIX concept; win32 gates on PATHEXT",
    )
    def test_a_non_executable_file_is_not_a_launcher(
        self, tmp_path: Path,
    ) -> None:
        (tmp_path / "aelf-hook").write_text("not executable", encoding="utf-8")
        assert setup_executable(tmp_path, "aelf-hook") is None


def setup_executable(directory: Path, name: str) -> Path | None:
    from aelfrice import setup

    return setup._executable_in_dir(directory, name)
