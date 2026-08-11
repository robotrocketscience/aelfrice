"""#1412: the shared console-launcher primitive.

Nobody on the team has a Windows host, so — following the pattern
`test_windows_portability_1329.py` established — every arm here drives the
Windows branch explicitly from POSIX CI rather than skipping. The `windows=`
keyword is the seam that makes that possible; `test_the_platform_flag_is_not
_bound_at_definition_time` is the arm that proves the seam is real, because
the natural implementation of it silently is not.
"""
from __future__ import annotations

import json
import os
import sys
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


def test_the_unbound_default_survives_into_the_hooks_transaction(
    tmp_path: Path,
) -> None:
    """The same guard, one layer up, on the path #1428 rewrote.

    `install_codex_hooks` and `remove_codex_hooks` stopped being the code
    that decides ownership: they are one-line delegations into
    `_commit_hooks_transaction`, and the decision moved into `_plan_install`
    / `_plan_remove`. A `windows` default bound at *either* layer reads the
    platform once at import, and the arm above keeps passing, because it
    never enters this path at all.

    Driven in two halves, because one probe cannot reach both layers here:

    * **The plan functions, under a patched `os.name`.** They are pure, so
      the patch touches nothing else. (`tempfile` joins with `ntpath` under
      `os.name == "nt"`, so the *committing* half cannot run under this
      patch on a POSIX build — `os.replace` fails and the transaction turns
      the OSError into `result.error`, which would make the assertion below
      vacuous rather than red.) `desired_codex_hooks` is frozen for the same
      class of reason: it reads `sysconfig` under the `nt` scheme, which is
      not what is being tested.
    * **The public delegations, driven at `_resolve_windows`.** This still
      distinguishes a bound default from a live seam: `_resolve_windows`
      only consults the platform when its argument is `None`, so a default
      bound at definition time would pass an explicit `False` down and the
      patch would be ignored — exactly the failure this arm exists to catch.
    """
    import aelfrice.session_ring  # noqa: F401 - pre-warm the lazy import
    from aelfrice import host_codex, launcher

    frozen = host_codex.desired_codex_hooks("user")
    hooks_path = tmp_path / "hooks.json"
    owned = json.dumps({"hooks": {"UserPromptSubmit": [
        {"hooks": [{"type": "command", "command": WIN_PATH}]},
    ]}}, indent=2) + "\n"

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "name", "nt")
        mp.setattr(
            host_codex, "desired_codex_hooks", lambda scope="user": frozen,
        )
        serialized, _ = host_codex._plan_install(
            owned, "user", False, hooks_path,
        )
        assert serialized is not None
        groups = json.loads(serialized)["hooks"]["UserPromptSubmit"]
        assert [
            h["command"] for g in groups for h in g["hooks"]
        ].count(WIN_PATH) == 0, groups

        serialized, result = host_codex._plan_remove(owned, hooks_path)
        assert result.changed is True
        assert serialized is not None
        assert "UserPromptSubmit" not in json.loads(serialized)["hooks"]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            launcher, "_resolve_windows", lambda w: True if w is None else w,
        )
        hooks_path.write_text(owned, encoding="utf-8")
        assert host_codex.remove_codex_hooks(hooks_path).changed is True
        assert json.loads(hooks_path.read_text(encoding="utf-8"))["hooks"] == {}


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

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            # A system interpreter installs here; the space is not exotic.
            (r"C:\Program Files\Python312\Scripts\aelf-hook.EXE", "aelf-hook"),
            # Any profile may carry one.
            (r"C:\Users\Ana Maria\.venv\Scripts\aelf-hook.exe", "aelf-hook"),
            # '#' is a legal path character, not a comment introducer.
            (r"C:\Users\dev#1\.venv\Scripts\aelf-hook.exe", "aelf-hook"),
            # Unquoted, with a trailing argument.
            (r"C:\Program Files\X\aelf-hook.exe --flag", "aelf-hook"),
        ],
    )
    def test_an_unquoted_spaced_path_still_yields_the_launcher_key(
        self, command: str, expected: str,
    ) -> None:
        """`_resolve_script` writes the resolved path unquoted.

        The quoted form was the only one covered, and nothing in this repo
        emits it, so the acceptance criterion "paths containing spaces are
        covered" was ticked on a form that never occurs. Whitespace-splitting
        an unquoted `C:\\Program Files\\...` command yields the program token
        `C:\\Program`, whose key is `program`, so our own handler stops being
        recognised as ours and #1412's whole symptom table reproduces.

        windows-smoke cannot catch this: the runner's paths have no spaces.
        """
        assert launcher.command_launcher_key(command, windows=True) == expected

    def test_an_argument_that_looks_like_a_launcher_is_not_the_program(
        self,
    ) -> None:
        """Rejoining tokens must not widen ownership.

        Ownership drives `remove_codex_hooks` and `prune_broken_aelf_hooks`,
        so reading an argument as the program is how you delete a file
        somebody else owns.
        """
        assert launcher.command_launcher_key(
            r"C:\Other\wrapper.exe --out aelf-hook.exe", windows=True,
        ) == "wrapper"

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

        Compared against the *pinned* scheme, not `sysconfig`'s default one.
        `get_path("scripts")` with no scheme resolves through
        `get_default_scheme()`, which a distributor can patch — Debian's
        `posix_local` is the common one — so asserting against it would fail
        here for a reason that has nothing to do with this code. The
        following test is the one that distinguishes the fix from the bug.
        """
        scheme = "nt" if os.name == "nt" else "posix_prefix"
        expected = sysconfig.get_path(
            "scripts", scheme, vars={
                "base": sys.prefix, "installed_base": sys.prefix,
                "platbase": sys.prefix, "installed_platbase": sys.prefix,
            },
        )
        assert launcher.scripts_dir() == Path(expected)

    def test_scripts_dir_follows_a_reassigned_prefix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """`sysconfig` caches its prefix variables at first import.

        A bare `sysconfig.get_path("scripts")` therefore does not follow a
        reassigned `sys.prefix`, so the whole venv branch silently reported
        the *installing* interpreter's directory. That version passed the
        entire suite locally and turned six tests red on CI. `scripts_dir`
        passes `sys.prefix` in explicitly and reads it per call; this is the
        arm that goes red if it stops doing so.
        """
        monkeypatch.setattr(sys, "prefix", str(tmp_path))
        assert launcher.scripts_dir().is_relative_to(tmp_path)

    def test_which_in_takes_an_explicit_directory(self, tmp_path: Path) -> None:
        target = _install_launcher(tmp_path)
        assert launcher.which_in(tmp_path, "aelf-hook") == target
        assert launcher.which_in(tmp_path, "aelf-missing") is None

    def test_which_on_path_passes_an_explicit_path(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression guard for the bare `shutil.which(name)` call.

        This pins the mechanism only. It passes on both the vulnerable and
        the fixed code, because passing `path=` does **not** by itself avoid
        the win32 curdir search — see
        `test_a_current_directory_hit_is_rejected_even_with_an_explicit_path`,
        which pins the property.
        """
        seen: list[object] = []

        def fake_which(name: str, mode: int = 1, path: object = None) -> None:
            seen.append(path)
            return None

        monkeypatch.setattr(launcher.shutil, "which", fake_which)
        monkeypatch.setenv("PATH", "/sentinel/bin")
        assert launcher.which_on_path("aelf-hook") is None
        assert seen == ["/sentinel/bin"]

    def test_a_current_directory_hit_is_rejected_even_with_an_explicit_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """`path=` does not suppress the win32 current-directory search.

        CPython inserts `os.curdir` at the front of the search list whenever
        the command has no directory part. That insertion is in the `else`
        branch of the dirname test, **not** under `if path is None` — so the
        earlier "passing `path=` stays out of that branch" reasoning was
        wrong and the fix built on it changed nothing. A stray
        `aelf-hook.exe` in the process's working directory still won and
        `_resolve_script` still pinned it into settings.json.

        Simulated rather than skipped, so it runs on POSIX CI: `shutil.which`
        is replaced by one that returns a curdir hit exactly as win32 would.
        """
        on_path = tmp_path / "real"
        on_path.mkdir()
        intruder = tmp_path / "cwd" / "aelf-hook.exe"
        intruder.parent.mkdir()
        intruder.write_text("", encoding="utf-8")

        monkeypatch.setattr(
            launcher.shutil, "which",
            lambda name, mode=1, path=None: str(intruder),
        )
        monkeypatch.setenv("PATH", str(on_path))
        assert launcher.which_on_path("aelf-hook") is None

        # ...and it is kept when the directory really is on PATH.
        monkeypatch.setenv("PATH", str(intruder.parent))
        assert launcher.which_on_path("aelf-hook") == intruder

    def test_which_in_does_not_escape_its_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        """A launcher in the working directory is not the venv's launcher.

        Trunk probed `directory / name` and could not escape. Delegating to
        `shutil.which` reintroduced an escape on win32, where `os.curdir` is
        inserted at the front of the search list regardless of `path=`,
        inverting the "project scope prefers the active environment over a
        global shim" acceptance criterion.

        This states the property but **cannot fail on POSIX CI**: the curdir
        insertion is win32-only, so the delegation this replaced passes here
        too. The test that actually goes red on a re-delegation is
        `test_which_in_searches_a_directory_named_with_the_path_separator`,
        which fails on any implementation that routes the directory through a
        `PATH`-shaped parameter. Kept as the statement of intent, with the
        enforcement named.
        """
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        _install_launcher(cwd)
        empty_scripts = tmp_path / "Scripts"
        empty_scripts.mkdir()

        monkeypatch.chdir(cwd)
        assert launcher.which_in(empty_scripts, "aelf-hook") is None

    def test_which_in_searches_a_directory_named_with_the_path_separator(
        self, tmp_path: Path,
    ) -> None:
        """`shutil.which` splits `path=` on `os.pathsep`; a directory is not a PATH.

        ``/home/a:b/.venv/bin`` is a legal POSIX path, and routing it through
        a PATH-shaped parameter silently searched nothing. The delegation
        therefore lost a resolution the plain probe found — a POSIX
        regression in a change whose stated scope was Windows-only.
        """
        odd = tmp_path / f"a{os.pathsep}b"
        odd.mkdir()
        target = _install_launcher(odd)
        assert launcher.which_in(odd, "aelf-hook") == target

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
