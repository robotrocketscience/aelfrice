"""#1412: `prune_broken_aelf_hooks` deleted working Windows installs.

The mechanism, in order:

1. `_inspect_command` tokenised with `shlex.split`, whose POSIX mode treats
   a backslash as an escape character. ``C:\\Scripts\\aelf-hook.exe`` became
   ``C:Scriptsaelf-hook.exe``.
2. That has no separator left in it, so the "is this a path?" test —
   ``"/" in program`` — was False and the mangled string went to the bare
   `$PATH` branch.
3. Nothing on `$PATH` is called ``C:Scriptsaelf-hook.exe``, so the hook was
   classified ``broken``.
4. `aelf setup` calls `prune_broken_aelf_hooks` unconditionally, so the next
   setup **deleted a correctly installed hook**.

Driving this from POSIX needs two things.

The `windows` fixture injects Windows semantics *at the launcher boundary*
rather than setting `os.name`. Patching `os.name` globally also reflavours
`pathlib.Path` into `WindowsPath`, so `tmp_path` starts rendering with
backslashes and the atomic settings write fails on its own temp file — the
test then fails for a reason that has nothing to do with the defect. Pinning
the boundary is also the stronger assertion: it fails if `doctor` stops
routing through the shared primitive, which is the regression that would
reintroduce the bug.

The `installed_hook` fixture creates a file whose *name literally contains
backslashes*, which is legal on POSIX, so the un-mangled path resolves to a
real file on disk and the mangled one does not. That is what makes the
assertion two-sided rather than "broken either way".

What cannot be checked from here is that a real Windows host agrees; the
`windows-smoke` job covers that.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice import launcher
from aelfrice.doctor import (
    _entry_is_broken_aelf_hook,
    _inspect_command,
    prune_broken_aelf_hooks,
)


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the Windows branch of every launcher call `doctor` makes."""
    tokens, key = launcher.command_tokens, launcher.command_launcher_key
    monkeypatch.setattr(
        launcher, "command_tokens",
        lambda c, *, windows=None: tokens(c, windows=True),
    )
    monkeypatch.setattr(
        launcher, "command_launcher_key",
        lambda c, *, windows=None: key(c, windows=True),
    )


@pytest.fixture
def installed_hook(tmp_path: Path) -> str:
    """A real file whose name carries the backslash the old code ate."""
    target = tmp_path / r"Scripts\aelf-hook.exe"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o755)
    command = str(target)
    assert "\\" in command
    assert Path(command).is_file()
    return command


def _settings(path: Path, command: str) -> Path:
    path.write_text(json.dumps({
        "hooks": {
            "UserPromptSubmit": [
                {"hooks": [{"type": "command", "command": command}]},
            ],
        },
    }), encoding="utf-8")
    return path


class TestInspection:
    def test_a_backslash_path_is_treated_as_a_path(
        self, tmp_path: Path, installed_hook: str, windows: None,
    ) -> None:
        finding = _inspect_command(tmp_path / "settings.json", "x", installed_hook)
        assert finding.status == "ok", finding.detail
        assert finding.program == installed_hook

    def test_the_old_tokeniser_would_have_called_it_broken(
        self, installed_hook: str,
    ) -> None:
        """Pins the mechanism, so the arm above cannot pass vacuously.

        If `shlex.split` ever stopped eating the backslash, the fix would be
        untested and this test says so.
        """
        import shlex

        mangled = shlex.split(installed_hook)[0]
        assert "\\" not in mangled
        assert not Path(mangled).exists()

    def test_a_genuinely_missing_windows_hook_is_still_broken(
        self, tmp_path: Path, windows: None,
    ) -> None:
        """The fix must not make prune blind — only accurate.

        The *detail* is the load-bearing half. A drive-letter path contains
        no forward slash, so `"/" in program` alone sent it to the bare-name
        `$PATH` branch and the diagnosis read ``'C:\\...\\aelf-hook.exe' not
        on $PATH`` — which sends the user looking at their PATH for a fault
        that is a missing file. Status is `broken` either way, so asserting
        only the status pins nothing.
        """
        finding = _inspect_command(
            tmp_path / "settings.json", "x",
            r"C:\Users\dev\.venv\Scripts\aelf-hook.exe",
        )
        assert finding.status == "broken"
        assert finding.detail == "path does not exist"
        assert "$PATH" not in (finding.detail or "")


class TestPrune:
    def test_prune_does_not_delete_an_installed_windows_hook(
        self, tmp_path: Path, installed_hook: str, windows: None,
    ) -> None:
        """The reported data loss, asserted end to end."""
        settings = _settings(tmp_path / "settings.json", installed_hook)
        result = prune_broken_aelf_hooks(settings)

        assert result.total_removed == 0, result.removed_per_event
        entries = json.loads(settings.read_text(encoding="utf-8"))
        assert entries["hooks"]["UserPromptSubmit"][0]["hooks"][0][
            "command"
        ] == installed_hook

    def test_prune_still_removes_a_hook_that_really_is_gone(
        self, tmp_path: Path, windows: None,
    ) -> None:
        settings = _settings(
            tmp_path / "settings.json",
            r"C:\Users\dev\.venv\Scripts\aelf-hook.exe",
        )
        assert prune_broken_aelf_hooks(settings).total_removed == 1

    def test_a_foreign_hook_is_never_pruned(
        self, tmp_path: Path, windows: None,
    ) -> None:
        settings = _settings(
            tmp_path / "settings.json", r"C:\Tools\conversation-logger.exe",
        )
        assert prune_broken_aelf_hooks(settings).total_removed == 0

    def test_the_broken_predicate_recognises_a_windows_launcher(
        self, tmp_path: Path, windows: None,
    ) -> None:
        """`aelf-hook.EXE` must read as an `aelf-*` entry at all.

        Under the old derivation the basename was the whole command string,
        so this returned False and the entry was merely skipped. Recognising
        it is what puts it in scope for prune — and therefore what makes the
        `_inspect_command` accuracy above load-bearing.
        """
        entry = {"hooks": [{
            "type": "command",
            "command": r"C:\Users\dev\.venv\Scripts\AELF-HOOK.EXE",
        }]}
        assert _entry_is_broken_aelf_hook(tmp_path / "settings.json", entry)


def test_posix_semantics_are_unchanged(tmp_path: Path) -> None:
    """On POSIX a backslash *is* an escape; nothing here may change that.

    Same fixture, no `windows` fixture: the command must still be read the
    way a POSIX shell would read it.
    """
    target = tmp_path / r"Scripts\aelf-hook.exe"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    finding = _inspect_command(tmp_path / "settings.json", "x", str(target))
    assert finding.status == "broken"
