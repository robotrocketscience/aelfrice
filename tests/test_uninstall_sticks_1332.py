"""#1332: `aelf uninstall` must actually stick.

Two independent defects, either of which alone reproduces the field report
that hooks kept injecting after a completed uninstall:

1. the teardown never wrote the auto-install opt-out, so the next `aelf`
   command of any kind silently reinstalled the whole manifest —
   `cli.main()` calls `auto_install_at_cli_entry` on every invocation;
2. it delegated with `scope="user"` hardcoded while `setup` auto-detects
   scope, so project-scope hooks survived a teardown reported as clean.

## Read this before adding a test here

**This module drives a destructive command.** A previous session pinned
`HOME`, `AELFRICE_DOTDIR`, `AELFRICE_DB` *and* `Path.home` and still had one
`pytest` run strip 24 hook entries from the developer's real
`~/.claude/settings.json` and write into their real
`~/.aelfrice/opt-out-hooks.json`. The escape routes are module constants
built from `Path.home()` at import: patching `Path.home` afterwards never
reaches them.

So the fixture below is a **tripwire**: it refuses to yield unless every
home-rooted path the uninstall path can reach resolves under `tmp_path`. A
missed constant is destructive rather than merely red, and the tripwire is
what converts it into a loud failure before any test body runs.
`test_the_tripwire_fails_closed` proves the tripwire itself works — without
that arm the sandbox is an assumption, and this is not a surface to assume
on.

CI cannot catch the destructive class either way: a runner's HOME is empty,
so the dangerous path is inert there. It is green-in-CI / destructive-locally
by construction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from aelfrice import auto_install, setup as setup_mod
from aelfrice.cli import _uninstall_settings_targets, build_parser


def _sandbox_paths() -> dict[str, Path]:
    """Every home-rooted path the uninstall path can reach.

    Resolved fresh on each call — the point is to observe what the code
    would use *now*, not what it was patched to.
    """
    return {
        "setup.USER_SETTINGS_PATH": setup_mod.USER_SETTINGS_PATH,
        "setup.default_settings_path('user')": setup_mod.default_settings_path(
            "user"
        ),
        "auto_install.OPT_OUT_PATH": auto_install.OPT_OUT_PATH,
        "auto_install.AELFRICE_DOTDIR": auto_install.AELFRICE_DOTDIR,
        "Path.home()": Path.home(),
    }


def _assert_all_sandboxed(root: Path) -> None:
    escaped = {
        name: path
        for name, path in _sandbox_paths().items()
        if not str(path.resolve()).startswith(str(root.resolve()))
    }
    if escaped:
        detail = "\n".join(f"  {n} -> {p}" for n, p in escaped.items())
        raise AssertionError(
            "REFUSING TO RUN — these resolve outside the sandbox and this "
            f"test drives a destructive command:\n{detail}\n"
            "Patch them before yielding; a miss here deletes real config."
        )


@pytest.fixture
def sandboxed_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """A fully isolated fake host, or no test at all.

    Patches the module constants rather than only `Path.home`, because the
    constants were bound at import and a `Path.home` patch does not reach
    them. Then asserts the result before yielding.
    """
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".aelfrice").mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("AELFRICE_DOTDIR", str(home / ".aelfrice"))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("AELFRICE_NO_AUTO_INSTALL", "1")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        setup_mod, "USER_SETTINGS_PATH", home / ".claude" / "settings.json"
    )
    monkeypatch.setattr(
        auto_install, "AELFRICE_DOTDIR", home / ".aelfrice"
    )
    monkeypatch.setattr(
        auto_install,
        "OPT_OUT_PATH",
        home / ".aelfrice" / "opt-out-hooks.json",
    )
    monkeypatch.setattr(
        auto_install, "STAMP_PATH", home / ".aelfrice" / "stamp.json"
    )

    _assert_all_sandboxed(tmp_path)
    return home


def test_the_tripwire_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The sandbox must be verified, not assumed.

    Patch four of the five and leave `auto_install.OPT_OUT_PATH` pointing
    at the real home; the tripwire has to notice. Without this arm a future
    refactor could silently stop covering a constant and every test here
    would still pass — while writing to a real config.
    """
    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(
        setup_mod, "USER_SETTINGS_PATH", home / ".claude" / "settings.json"
    )
    monkeypatch.setattr(auto_install, "AELFRICE_DOTDIR", home / ".aelfrice")
    # OPT_OUT_PATH deliberately left pointing at the developer's real home.
    with pytest.raises(AssertionError, match="REFUSING TO RUN"):
        _assert_all_sandboxed(tmp_path)


def test_the_tripwire_passes_when_fully_sandboxed(
    sandboxed_host: Path, tmp_path: Path
) -> None:
    """The positive control for the arm above."""
    _assert_all_sandboxed(tmp_path)
    assert sandboxed_host.is_dir()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOOK_SETTINGS: dict[str, Any] = {
    "hooks": {
        "UserPromptSubmit": [
            {"hooks": [{"type": "command", "command": "aelf-hook"}]}
        ]
    },
    "somethingElse": {"keepMe": True},
}


def _write_settings(path: Path, payload: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload if payload is not None else _HOOK_SETTINGS))


def _uninstall_args(**overrides: Any) -> argparse.Namespace:
    """Build the namespace through the real parser.

    Synthesising one by hand omits attributes the CLI always supplies and
    the tests then die on an `AttributeError` the product cannot raise.
    """
    parser = build_parser()
    args = parser.parse_args(["uninstall", "--keep-db"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# ---------------------------------------------------------------------------
# AC2 — scope discovery
# ---------------------------------------------------------------------------


class TestSettingsTargetDiscovery:
    def test_both_scopes_are_returned_when_both_exist(
        self, sandboxed_host: Path, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """The defect: uninstall hardcoded user scope while setup
        auto-detects, so project hooks outlived a 'successful' teardown."""
        project = tmp_path / "proj"
        _write_settings(project / ".claude" / "settings.json")
        _write_settings(sandboxed_host / ".claude" / "settings.json")
        monkeypatch.chdir(project)

        targets = _uninstall_settings_targets(_uninstall_args())

        assert sandboxed_host / ".claude" / "settings.json" in targets
        assert project / ".claude" / "settings.json" in targets

    def test_a_missing_scope_is_skipped(
        self, sandboxed_host: Path, tmp_path: Path, monkeypatch: Any
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        _write_settings(sandboxed_host / ".claude" / "settings.json")
        monkeypatch.chdir(project)

        targets = _uninstall_settings_targets(_uninstall_args())

        assert targets == [sandboxed_host / ".claude" / "settings.json"]

    def test_one_file_reachable_two_ways_is_cleaned_once(
        self, sandboxed_host: Path, monkeypatch: Any
    ) -> None:
        """A project rooted at HOME resolves both scopes to one file.
        Unsetting it twice prints 'no matching hook' on the second pass,
        which reads to a user as the teardown having failed."""
        _write_settings(sandboxed_host / ".claude" / "settings.json")
        monkeypatch.chdir(sandboxed_host)

        targets = _uninstall_settings_targets(_uninstall_args())

        assert len(targets) == 1, targets

    def test_user_scope_is_attempted_even_when_the_file_is_absent(
        self, sandboxed_host: Path, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """`_cmd_unsetup` also removes the statusline snippet, so skipping
        the call when settings.json happens to be missing would silently
        skip that too. This is the pre-#1332 behaviour and the change is
        meant to widen the teardown, not narrow a path that worked.

        Caught by `test_uninstall_removes_the_rebuilder_hook` when an
        earlier cut of this filtered both scopes on existence.
        """
        user_settings = sandboxed_host / ".claude" / "settings.json"
        assert not user_settings.exists()
        monkeypatch.chdir(tmp_path)

        targets = _uninstall_settings_targets(_uninstall_args())

        assert targets == [user_settings]

    def test_project_scope_is_filtered_on_existence(
        self, sandboxed_host: Path, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """The other half of the asymmetry: inventing a settings.json in
        whatever directory the user ran from is not teardown."""
        project = tmp_path / "proj"
        project.mkdir()
        monkeypatch.chdir(project)

        targets = _uninstall_settings_targets(_uninstall_args())

        assert project / ".claude" / "settings.json" not in targets

    def test_an_explicit_settings_path_is_never_widened(
        self, sandboxed_host: Path, tmp_path: Path
    ) -> None:
        """The user named a file. Quietly editing a second one is not what
        they asked for, even though the default now covers both scopes."""
        chosen = tmp_path / "custom" / "settings.json"
        _write_settings(chosen)
        _write_settings(sandboxed_host / ".claude" / "settings.json")

        targets = _uninstall_settings_targets(
            _uninstall_args(settings_path=str(chosen))
        )

        assert targets == [chosen]


# ---------------------------------------------------------------------------
# AC1 — the teardown must survive the next `aelf` command
# ---------------------------------------------------------------------------


class TestUninstallWritesTheOptOut:
    """The field report: hooks kept injecting after a completed uninstall.

    `cli.main()` calls `auto_install_at_cli_entry` on *every* invocation,
    and it returns early only on the env opt-out, the host opt-out, or a
    non-`uv tool` install. Uninstall set none of them, so the next `aelf`
    command silently reinstalled the manifest.
    """

    def _run_uninstall(self, out: Any) -> int:
        from aelfrice.cli import _cmd_uninstall

        return _cmd_uninstall(_uninstall_args(yes=True), out)

    def test_the_host_opt_out_is_written(
        self, sandboxed_host: Path, monkeypatch: Any
    ) -> None:
        import io

        _write_settings(sandboxed_host / ".claude" / "settings.json")
        monkeypatch.chdir(sandboxed_host)
        assert "claude" not in auto_install.read_host_opt_outs()

        self._run_uninstall(io.StringIO())

        assert "claude" in auto_install.read_host_opt_outs(), (
            "without this the next `aelf` command reinstalls everything"
        )

    def test_auto_install_then_declines_to_run(
        self, sandboxed_host: Path, monkeypatch: Any
    ) -> None:
        """The assertion that matters is about the *next command*, not the
        marker file. Drive the real entry point and assert it no-ops."""
        import io

        _write_settings(sandboxed_host / ".claude" / "settings.json")
        monkeypatch.chdir(sandboxed_host)
        self._run_uninstall(io.StringIO())

        called: list[str] = []
        monkeypatch.setattr(
            auto_install,
            "maybe_install_manifest",
            lambda **kw: called.append("ran"),
        )
        # The env guard would mask the thing under test.
        monkeypatch.delenv("AELFRICE_NO_AUTO_INSTALL", raising=False)
        monkeypatch.setattr(
            auto_install, "is_running_from_uv_tool_install", lambda: True
        )

        auto_install.auto_install_at_cli_entry(installed_version="9.9.9")

        assert called == [], (
            "auto-install ran after uninstall — the teardown is not durable"
        )

    def test_keep_hook_does_not_write_the_opt_out(
        self, sandboxed_host: Path, monkeypatch: Any
    ) -> None:
        """`--keep-hook` means the user is keeping the hooks. Opting the
        host out would then block the updates that keep those hooks
        working, which is not what they asked for."""
        import io
        from aelfrice.cli import _cmd_uninstall

        _write_settings(sandboxed_host / ".claude" / "settings.json")
        monkeypatch.chdir(sandboxed_host)

        _cmd_uninstall(_uninstall_args(yes=True, keep_hook=True), io.StringIO())

        assert "claude" not in auto_install.read_host_opt_outs()


# ---------------------------------------------------------------------------
# AC2 (end to end) + AC3 — both scopes cleaned, and the output says so
# ---------------------------------------------------------------------------


class TestUninstallCleansEveryScope:
    def test_project_scope_hooks_do_not_survive(
        self, sandboxed_host: Path, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """The second defect, end to end. A user who installed project
        scope was told the teardown succeeded while the hooks kept firing
        on every prompt in that project."""
        import io
        from aelfrice.cli import _cmd_uninstall

        project = tmp_path / "proj"
        project_settings = project / ".claude" / "settings.json"
        _write_settings(project_settings)
        _write_settings(sandboxed_host / ".claude" / "settings.json")
        monkeypatch.chdir(project)

        _cmd_uninstall(_uninstall_args(yes=True), io.StringIO())

        remaining = json.loads(project_settings.read_text())
        hooks = remaining.get("hooks", {}).get("UserPromptSubmit", [])
        commands = [
            h.get("command", "")
            for group in hooks
            for h in group.get("hooks", [])
        ]
        assert not any("aelf" in c for c in commands), commands

    def test_unrelated_settings_keys_are_preserved(
        self, sandboxed_host: Path, monkeypatch: Any
    ) -> None:
        """Widening the blast radius to a second file makes this arm more
        important, not less: uninstall now writes files it did not before."""
        import io
        from aelfrice.cli import _cmd_uninstall

        settings = sandboxed_host / ".claude" / "settings.json"
        _write_settings(settings)
        monkeypatch.chdir(sandboxed_host)

        _cmd_uninstall(_uninstall_args(yes=True), io.StringIO())

        assert json.loads(settings.read_text())["somethingElse"] == {
            "keepMe": True
        }

    def test_the_output_names_what_is_left(
        self, sandboxed_host: Path, monkeypatch: Any
    ) -> None:
        """AC3. A user who followed this command's own output had no way
        to learn the package was still installed and hooks could return."""
        import io
        from aelfrice.cli import _cmd_uninstall

        _write_settings(sandboxed_host / ".claude" / "settings.json")
        monkeypatch.chdir(sandboxed_host)
        out = io.StringIO()

        _cmd_uninstall(_uninstall_args(yes=True), out)
        text = out.getvalue()

        assert "still installed" in text
        assert "uv tool uninstall" in text, (
            "uv tool is the only supported channel (#730); the old message "
            "said pip"
        )
        assert "hook cleanup ran on:" in text
