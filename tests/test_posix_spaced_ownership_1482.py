"""#1482: hook ownership under a POSIX install path containing a space.

#1412 fixed this for Windows only. `setup._resolve_script` writes the
resolved path **unquoted** on both platforms, so a space anywhere in it
splits the command; the Windows rejoin keys on a launcher suffix (`.exe`,
`.cmd`, ...) that POSIX console scripts never carry, so on POSIX
``/home/first last/.venv/bin/aelf-hook`` still keyed as ``first`` and our
own handler stopped being recognised as ours. macOS builds home directories
from the user's full name, so this is a routine path, not an exotic one.

Every arm drives `windows=False` explicitly, or a fixture that pins the
install path — the defect is invisible to a runner whose own paths have no
spaces, which is why `windows-smoke` never saw it and why the POSIX CI job
never saw it either.

Two properties are load-bearing here and each has its own arm:

*No widening.* Ownership drives `remove_codex_hooks`, `prune_broken_aelf
_hooks` and the claude-host uninstall, so claiming more than we own deletes
somebody else's entry. A foreign handler at a spaced path, an absolute-path
argument, and a bare relative argument are all asserted **unclaimed**.

*No filesystem.* The candidate scan is a string rule on purpose. An
existence probe would answer correctly today and wrongly the moment the venv
is deleted — which is the exact state `aelf unsetup` runs in. One arm removes
the launchers from disk and asserts uninstall still recognises its own
entries.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice import launcher, setup
from aelfrice.doctor import (
    _entry_is_broken_aelf_hook,
    find_duplicate_hook_entries,
    prune_broken_aelf_hooks,
)
from aelfrice.host_codex import (
    claude_host_has_aelfrice_hooks,
    desired_codex_hooks,
    doctor_codex,
    install_codex_hooks,
    remove_codex_hooks,
)

SPACED = "first last"
FOREIGN = "foreign-hook"


def _handler(command: str) -> dict[str, object]:
    return {"type": "command", "command": command}


def _read_hooks(hooks_path: Path) -> dict[str, object]:
    doc = json.loads(hooks_path.read_text(encoding="utf-8"))
    return doc["hooks"]


def _commands_in(hooks: dict[str, object]) -> list[str]:
    return [
        h["command"]
        for groups in hooks.values()
        for g in groups  # type: ignore[union-attr]
        for h in g["hooks"]
    ]


@pytest.fixture
def spaced_bin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Resolve every hook script under a directory whose name has a space.

    Patches `setup._resolve_script`, the one function that decides what path
    is written into a hook command, so the whole installer stack below runs
    against the reported install shape.
    """
    bin_dir = tmp_path / SPACED / "bin"
    bin_dir.mkdir(parents=True)

    def _resolve(script_name: str, scope: str) -> str:
        target = bin_dir / script_name
        if not target.exists():
            target.write_text("#!/bin/sh\n", encoding="utf-8")
            target.chmod(0o755)
        return str(target)

    monkeypatch.setattr(setup, "_resolve_script", _resolve)
    assert " " in setup.resolve_hook_command("user")
    return bin_dir


def _desired_handler_count() -> int:
    return sum(
        len(group["hooks"])  # type: ignore[arg-type]
        for groups in desired_codex_hooks().values()
        for group in groups
    )


class TestKeyDerivation:
    def test_a_spaced_path_yields_the_console_script_name(self) -> None:
        """The reported defect, at the primitive."""
        cmd = "/home/first last/.venv/bin/aelf-hook"
        assert launcher.command_launcher_key(cmd, windows=False) == "first"
        assert launcher.command_program_keys(cmd, windows=False) == [
            "first", "aelf-hook",
        ]

    def test_an_interior_token_without_a_separator_is_scanned_over(
        self,
    ) -> None:
        """``/Users/First Middle Last/...`` — two spaces, one path."""
        assert launcher.command_program_keys(
            "/Users/First Middle Last/.venv/bin/aelf-hook", windows=False,
        ) == ["First", "aelf-hook"]

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            # An absolute-path argument stops the scan ('/' prefix).
            ("/opt/x/wrapper /usr/bin/aelf-hook", ["wrapper"]),
            # A switch stops it outright.
            ("/tmp/x/wrapper --out /tmp/y/aelf-hook", ["wrapper"]),
            # A bare relative argument carries no separator, so it is never
            # the end of a split path and is not emitted as a candidate.
            ("/usr/bin/env aelf-hook", ["env"]),
        ],
    )
    def test_an_argument_is_never_read_as_the_program(
        self, command: str, expected: list[str],
    ) -> None:
        """The no-widening rule, at the primitive.

        Ownership drives deletion; reading an argument as the program is how
        a fix for this issue would delete a file somebody else owns.
        """
        assert launcher.command_program_keys(
            command, windows=False,
        ) == expected

    def test_a_foreign_spaced_path_is_still_foreign(self) -> None:
        cmd = f"/home/{SPACED}/bin/{FOREIGN}"
        assert launcher.command_program_keys(cmd, windows=False) == [
            "first", FOREIGN,
        ]

    def test_windows_still_yields_exactly_one_key(self) -> None:
        """AC5: the Windows rejoin already resolves the program there.

        Offering candidates on Windows would widen it — ``wrapper.exe --out
        aelf-hook.exe`` is pinned by
        `test_windows_launcher_1412.py::test_an_argument_that_looks_like_a
        _launcher_is_not_the_program`, and this keeps that arm's subject.
        """
        for cmd in (
            r"C:\Program Files\X\aelf-hook.exe --flag",
            r"C:\Users\dev\.venv\Scripts\aelf-hook.EXE",
        ):
            assert launcher.command_program_keys(cmd, windows=True) == [
                "aelf-hook",
            ]
        assert launcher.command_program_keys(
            r"C:\Other\wrapper.exe --out aelf-hook.exe", windows=True,
        ) == ["wrapper"]

    def test_the_windows_gate_is_what_keeps_an_argument_out(self) -> None:
        """Where the platform gate is load-bearing rather than decorative.

        Windows accepts forward slashes, so an argument spelled as a
        forward-slash path is not switch-shaped and *does* carry a
        separator: the POSIX rule would read it as the continuation of the
        program path and claim ``aelf-hook``. On Windows the suffix rejoin
        already resolved the program from ``wrapper.exe``, so nothing is
        gained by scanning and a deletion is risked.
        """
        assert launcher.command_program_keys(
            r"C:\Tools\wrapper.exe C:/other/aelf-hook.exe", windows=True,
        ) == ["wrapper"]


class TestCodexSetup:
    def test_setup_twice_is_byte_identical(
        self, tmp_path: Path, spaced_bin: Path,
    ) -> None:
        """AC1."""
        hooks_path = tmp_path / "hooks.json"
        first = install_codex_hooks(hooks_path, windows=False)
        first_bytes = hooks_path.read_bytes()
        second = install_codex_hooks(hooks_path, windows=False)

        assert first.changed is True
        assert first.error is None
        assert second.changed is False
        assert hooks_path.read_bytes() == first_bytes
        counts = {e: len(v) for e, v in _read_hooks(hooks_path).items()}  # type: ignore[arg-type]
        assert set(counts.values()) == {1}, counts

    def test_a_third_run_does_not_accumulate_handlers(
        self, tmp_path: Path, spaced_bin: Path,
    ) -> None:
        """The field symptom: a duplicate group per event on every run."""
        hooks_path = tmp_path / "hooks.json"
        for _ in range(3):
            install_codex_hooks(hooks_path, windows=False)
        commands = _commands_in(_read_hooks(hooks_path))
        assert len(commands) == _desired_handler_count()
        assert len(set(commands)) == len(
            {
                setup.resolve_hook_command("user"),
                setup.resolve_transcript_logger_command("user"),
                setup.resolve_session_start_hook_command("user"),
                setup.resolve_stop_hook_command("user"),
                setup.resolve_search_tool_bash_command("user"),
                setup.resolve_pre_issue_guard_command("user"),
                setup.resolve_commit_ingest_command("user"),
            },
        )


class TestCodexDoctor:
    def test_every_owned_handler_is_counted(
        self, tmp_path: Path, spaced_bin: Path,
    ) -> None:
        """AC2, as an exact count.

        ``> 0`` would pass on a widened fix that also claimed the foreign
        handler planted beside ours.
        """
        hooks_path = tmp_path / "hooks.json"
        install_codex_hooks(hooks_path, windows=False)
        hooks = _read_hooks(hooks_path)
        groups = hooks["Stop"]
        groups[0]["hooks"].append(  # type: ignore[index,union-attr]
            _handler(str(spaced_bin / FOREIGN)),
        )
        hooks_path.write_text(
            json.dumps({"hooks": hooks}, indent=2) + "\n", encoding="utf-8",
        )

        report = doctor_codex(tmp_path, windows=False)
        assert report.owned_handler_count == _desired_handler_count()
        assert report.missing_events == []
        assert report.stale_commands == []

    def test_a_partially_removed_event_is_still_reported(
        self, tmp_path: Path, spaced_bin: Path,
    ) -> None:
        """#1430's check must survive the spaced path, not just pass it.

        Under the first-token key every handler of this install keyed to
        ``first``, so the expected and found key sets both collapsed to one
        element and a deleted handler was invisible.
        """
        hooks_path = tmp_path / "hooks.json"
        install_codex_hooks(hooks_path, windows=False)
        hooks = _read_hooks(hooks_path)
        handlers = hooks["UserPromptSubmit"][0]["hooks"]  # type: ignore[index]
        assert len(handlers) == 2
        del handlers[-1]
        hooks_path.write_text(
            json.dumps({"hooks": hooks}, indent=2) + "\n", encoding="utf-8",
        )

        report = doctor_codex(tmp_path, windows=False)
        assert report.missing_handlers == [
            "UserPromptSubmit:aelf-transcript-logger",
        ]


class TestCodexUnsetup:
    def test_unsetup_removes_every_group_it_installed(
        self, tmp_path: Path, spaced_bin: Path,
    ) -> None:
        """AC3."""
        hooks_path = tmp_path / "hooks.json"
        install_codex_hooks(hooks_path, windows=False)
        result = remove_codex_hooks(hooks_path, windows=False)

        assert result.changed is True
        assert _read_hooks(hooks_path) == {}

    def test_a_foreign_spaced_handler_survives(
        self, tmp_path: Path, spaced_bin: Path,
    ) -> None:
        """AC4: the no-widening rule, on the deletion path.

        The foreign program sits in the *same* spaced directory as ours, so
        it shares our first-token key. Any fix that decides ownership from a
        recovered path without re-checking the owned set deletes it.
        """
        hooks_path = tmp_path / "hooks.json"
        install_codex_hooks(hooks_path, windows=False)
        hooks = _read_hooks(hooks_path)
        foreign = str(spaced_bin / FOREIGN)
        hooks["Stop"][0]["hooks"].append(_handler(foreign))  # type: ignore[index,union-attr]
        hooks_path.write_text(
            json.dumps({"hooks": hooks}, indent=2) + "\n", encoding="utf-8",
        )

        remove_codex_hooks(hooks_path, windows=False)
        surviving = _read_hooks(hooks_path)
        assert surviving == {"Stop": [{"hooks": [_handler(foreign)]}]}

    def test_a_foreign_only_file_is_left_byte_identical(
        self, tmp_path: Path,
    ) -> None:
        """Every command here is spaced, and none of it is ours."""
        hooks_path = tmp_path / "hooks.json"
        hooks_path.write_text(json.dumps({"hooks": {
            "Stop": [{"hooks": [
                _handler(f"/opt/{SPACED}/bin/{FOREIGN}"),
                _handler("/opt/x/wrapper --out /opt/y/aelf-hook"),
                _handler("/usr/bin/env aelf-hook"),
            ]}],
        }}, indent=2) + "\n", encoding="utf-8")
        before = hooks_path.read_bytes()

        result = remove_codex_hooks(hooks_path, windows=False)
        assert result.changed is False
        assert hooks_path.read_bytes() == before

    def test_ownership_does_not_depend_on_the_launcher_on_disk(
        self, tmp_path: Path, spaced_bin: Path,
    ) -> None:
        """The reason the recovery is a string rule and not a stat.

        `aelf unsetup` is routinely run *after* the venv is gone. An
        existence-probe fix would pass every other arm in this file and fail
        exactly here, leaving the hooks wired forever.
        """
        hooks_path = tmp_path / "hooks.json"
        install_codex_hooks(hooks_path, windows=False)
        for stale in spaced_bin.iterdir():
            stale.unlink()
        spaced_bin.rmdir()

        assert remove_codex_hooks(hooks_path, windows=False).changed is True
        assert _read_hooks(hooks_path) == {}


class TestClaudeHost:
    """The issue's out-of-scope question, answered: the derivation is shared.

    `setup._command_basename` / `_entry_matches_basename` route through the
    same launcher key, so `--host claude` fails on a spaced install too —
    and worse than by duplication.
    """

    def test_two_hooks_in_one_event_do_not_replace_each_other(
        self, tmp_path: Path, spaced_bin: Path,
    ) -> None:
        """Both `aelf-hook` and `aelf-transcript-logger` land in one event.

        Under the first-token key both dedupe to ``first``, so installing
        the logger *replaced* the prompt hook: UserPromptSubmit ended up
        with one entry, and prompt-side retrieval was silently not wired.
        """
        settings = tmp_path / "settings.json"
        setup.install_user_prompt_submit_hook(
            settings, command=setup.resolve_hook_command("user"),
        )
        setup.install_transcript_ingest_hooks(
            settings, command=setup.resolve_transcript_logger_command("user"),
        )
        data = json.loads(settings.read_text(encoding="utf-8"))
        entries = data["hooks"]["UserPromptSubmit"]
        commands = [e["hooks"][0]["command"] for e in entries]
        assert len(entries) == 2, commands
        assert sorted(Path(c).name for c in commands) == [
            "aelf-hook", "aelf-transcript-logger",
        ]

    def test_reinstalling_the_same_hook_is_a_no_op(
        self, tmp_path: Path, spaced_bin: Path,
    ) -> None:
        settings = tmp_path / "settings.json"
        command = setup.resolve_hook_command("user")
        setup.install_user_prompt_submit_hook(settings, command=command)
        before = settings.read_bytes()
        second = setup.install_user_prompt_submit_hook(
            settings, command=command,
        )
        assert second.already_present is True
        assert settings.read_bytes() == before

    def test_uninstall_by_basename_finds_a_spaced_entry(
        self, tmp_path: Path, spaced_bin: Path,
    ) -> None:
        settings = tmp_path / "settings.json"
        setup.install_user_prompt_submit_hook(
            settings, command=setup.resolve_hook_command("user"),
        )
        result = setup.uninstall_user_prompt_submit_hook(
            settings, command_basename="aelf-hook",
        )
        assert result.removed == 1

    def test_uninstall_by_basename_leaves_a_foreign_spaced_entry(
        self, tmp_path: Path,
    ) -> None:
        """AC4 on the Claude-host uninstaller."""
        settings = tmp_path / "settings.json"
        foreign = f"/opt/{SPACED}/bin/{FOREIGN}"
        settings.write_text(json.dumps({"hooks": {
            "UserPromptSubmit": [{"hooks": [_handler(foreign)]}],
        }}, indent=2) + "\n", encoding="utf-8")
        before = settings.read_bytes()

        result = setup.uninstall_user_prompt_submit_hook(
            settings, command_basename="aelf-hook",
        )
        assert result.removed == 0
        assert settings.read_bytes() == before

    def test_the_dual_host_probe_sees_a_spaced_install(
        self, tmp_path: Path,
    ) -> None:
        """A silent one: a dual-host machine read as Codex-only, and
        `aelf setup --host codex` wrote the Claude-host auto-install opt-out
        over a live install."""
        settings = tmp_path / "settings.json"
        settings.write_text(json.dumps({"hooks": {
            "UserPromptSubmit": [
                {"hooks": [_handler(f"/home/{SPACED}/.venv/bin/aelf-hook")]},
            ],
        }}, indent=2) + "\n", encoding="utf-8")
        assert claude_host_has_aelfrice_hooks(settings, windows=False) is True

    def test_the_dual_host_probe_ignores_a_foreign_spaced_entry(
        self, tmp_path: Path,
    ) -> None:
        settings = tmp_path / "settings.json"
        settings.write_text(json.dumps({"hooks": {
            "UserPromptSubmit": [
                {"hooks": [_handler(f"/home/{SPACED}/.venv/bin/{FOREIGN}")]},
            ],
        }}, indent=2) + "\n", encoding="utf-8")
        assert claude_host_has_aelfrice_hooks(settings, windows=False) is False


class TestDoctorSettingsScans:
    def _settings(self, path: Path, commands: list[str]) -> Path:
        path.write_text(json.dumps({"hooks": {
            "UserPromptSubmit": [
                {"hooks": [_handler(c)]} for c in commands
            ],
        }}, indent=2) + "\n", encoding="utf-8")
        return path

    def test_a_duplicated_spaced_hook_is_reported(
        self, tmp_path: Path,
    ) -> None:
        cmd = f"/home/{SPACED}/.venv/bin/aelf-hook"
        settings = self._settings(tmp_path / "settings.json", [cmd, cmd])
        found = find_duplicate_hook_entries(settings)
        assert len(found) == 1
        assert found[0].basenames == "aelf-hook"
        assert found[0].count == 2

    def test_two_different_spaced_hooks_are_not_a_duplicate(
        self, tmp_path: Path,
    ) -> None:
        """The distinguishing arm for the collapse key.

        Both commands share the first token, so a fix that keyed the
        collapse on any recovered candidate — or on the fragment — reports
        a duplicate here and `--prune` would delete a live hook.
        """
        settings = self._settings(tmp_path / "settings.json", [
            f"/home/{SPACED}/.venv/bin/aelf-hook",
            f"/home/{SPACED}/.venv/bin/aelf-stop-hook",
        ])
        assert find_duplicate_hook_entries(settings) == []

    def test_a_foreign_spaced_duplicate_is_not_ours_to_report(
        self, tmp_path: Path,
    ) -> None:
        cmd = f"/home/{SPACED}/bin/{FOREIGN}"
        settings = self._settings(tmp_path / "settings.json", [cmd, cmd])
        assert find_duplicate_hook_entries(settings) == []

    def test_a_broken_spaced_hook_is_prunable(self, tmp_path: Path) -> None:
        settings = tmp_path / "settings.json"
        entry = {"hooks": [_handler(
            str(tmp_path / SPACED / "bin" / "aelf-hook"),
        )]}
        assert _entry_is_broken_aelf_hook(settings, entry) is True

    def test_a_broken_foreign_spaced_hook_is_not_prunable(
        self, tmp_path: Path,
    ) -> None:
        """Prune deletes what this predicate claims; it must claim only ours."""
        settings = tmp_path / "settings.json"
        entry = {"hooks": [_handler(
            str(tmp_path / SPACED / "bin" / FOREIGN),
        )]}
        assert _entry_is_broken_aelf_hook(settings, entry) is False

    def test_a_healthy_spaced_hook_is_not_prunable(
        self, tmp_path: Path, spaced_bin: Path,
    ) -> None:
        settings = tmp_path / "settings.json"
        entry = {"hooks": [_handler(setup.resolve_hook_command("user"))]}
        assert _entry_is_broken_aelf_hook(settings, entry) is False

    def test_prune_does_not_delete_a_healthy_spaced_install(
        self, tmp_path: Path, spaced_bin: Path,
    ) -> None:
        """The regression teaching prune to see spaced paths would cause.

        `_inspect_command` checked `tokens[0]`, so a healthy hook under a
        spaced path read as "path does not exist". That was inert only
        because the `aelf-*` predicate above could not recognise the entry
        either; fixing one without the other turns `aelf setup` — which runs
        prune unconditionally — into a deleter of working installs.
        """
        settings = self._settings(
            tmp_path / "settings.json",
            [setup.resolve_hook_command("user")],
        )
        before = settings.read_bytes()
        result = prune_broken_aelf_hooks(settings)
        assert result.total_removed == 0
        assert settings.read_bytes() == before

    def test_prune_removes_a_broken_spaced_install(
        self, tmp_path: Path,
    ) -> None:
        """And the other direction: a spaced path that really is gone."""
        settings = self._settings(
            tmp_path / "settings.json",
            [str(tmp_path / SPACED / "bin" / "aelf-hook")],
        )
        result = prune_broken_aelf_hooks(settings)
        assert result.total_removed == 1
        data = json.loads(settings.read_text(encoding="utf-8"))
        assert data["hooks"].get("UserPromptSubmit", []) == []

    def test_prune_leaves_a_broken_foreign_spaced_entry(
        self, tmp_path: Path,
    ) -> None:
        settings = self._settings(
            tmp_path / "settings.json",
            [str(tmp_path / SPACED / "bin" / FOREIGN)],
        )
        before = settings.read_bytes()
        assert prune_broken_aelf_hooks(settings).total_removed == 0
        assert settings.read_bytes() == before
