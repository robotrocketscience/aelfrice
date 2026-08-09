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
        target = tmp_path / "aelf-hook"
        target.write_text("#!/bin/sh\n", encoding="utf-8")
        target.chmod(0o755)
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
