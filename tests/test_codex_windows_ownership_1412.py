"""#1412: Codex hook ownership under Windows launcher spellings.

The field report: `aelf setup --host codex` on native Windows appended
another copy of all seven hook groups on every run, `aelf doctor --host
codex` reported zero handlers and issued no warning, and unsetup left the
originals behind. One root cause — ownership was decided from
``Path(cmd.split()[0]).name`` against a suffixless, case-sensitive set, so
``C:\\...\\Scripts\\aelf-hook.EXE`` was never ours.

Every arm drives `windows=True` explicitly. That keyword is the seam #1412
had to add: before it, `install_codex_hooks`/`remove_codex_hooks`/
`doctor_codex` took no platform flag at all, so the obvious regression tests
("install twice, assert not changed") took the POSIX branch on CI and passed
identically on a broken tree. Each `windows=True` arm below is paired with a
`windows=False` arm asserting POSIX behaviour did not move.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.host_codex import (
    desired_codex_hooks,
    doctor_codex,
    install_codex_hooks,
    remove_codex_hooks,
)

WIN_HOOK = r"C:\Users\dev\.venv\Scripts\aelf-hook.EXE"
WIN_LOGGER = r"C:\Users\dev\.venv\Scripts\aelf-transcript-logger.EXE"
FOREIGN = r"C:\Tools\foreign-hook.EXE"


def _handler(command: str) -> dict[str, object]:
    return {"type": "command", "command": command}


def _write(path: Path, hooks: dict[str, object]) -> None:
    path.write_text(json.dumps({"hooks": hooks}, indent=2) + "\n", "utf-8")


def _read(path: Path) -> dict[str, object]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc["hooks"]


class TestIdempotence:
    def test_setup_twice_is_byte_identical_under_windows(
        self, tmp_path: Path,
    ) -> None:
        """AC: a second `aelf setup --host codex` leaves hooks.json alone."""
        hooks_path = tmp_path / "hooks.json"
        first = install_codex_hooks(hooks_path, windows=True)
        first_bytes = hooks_path.read_bytes()
        second = install_codex_hooks(hooks_path, windows=True)

        assert first.changed is True
        assert second.changed is False
        assert hooks_path.read_bytes() == first_bytes
        counts = {e: len(v) for e, v in _read(hooks_path).items()}  # type: ignore[arg-type]
        assert set(counts.values()) == {1}, counts

    def test_a_windows_launcher_spelling_is_replaced_not_appended(
        self, tmp_path: Path,
    ) -> None:
        """The reported failure, reproduced from its own artefacts.

        A hooks.json already carrying `.EXE` launchers under `Scripts` — what
        a real Windows install writes — must be recognised as ours and
        replaced, not duplicated.
        """
        hooks_path = tmp_path / "hooks.json"
        _write(hooks_path, {
            "UserPromptSubmit": [
                {"hooks": [_handler(WIN_HOOK), _handler(WIN_LOGGER)]},
            ],
        })
        install_codex_hooks(hooks_path, windows=True)

        groups = _read(hooks_path)["UserPromptSubmit"]
        assert len(groups) == 1, groups  # not 2
        assert all(
            WIN_HOOK not in json.dumps(g) for g in groups  # type: ignore[arg-type]
        )

    def test_the_same_file_still_duplicates_under_posix_rules(
        self, tmp_path: Path,
    ) -> None:
        """The distinguishing arm: POSIX must *not* claim a Windows path.

        Without this, `test_a_windows_launcher_spelling_is_replaced_not
        _appended` would pass on a tree where the platform gate was removed
        and normalisation applied unconditionally. Ownership drives deletion,
        so claiming more than we own is the dangerous direction.
        """
        hooks_path = tmp_path / "hooks.json"
        _write(hooks_path, {
            "UserPromptSubmit": [{"hooks": [_handler(WIN_HOOK)]}],
        })
        install_codex_hooks(hooks_path, windows=False)
        groups = _read(hooks_path)["UserPromptSubmit"]
        assert len(groups) == 2, groups
        assert groups[0] == {"hooks": [_handler(WIN_HOOK)]}  # type: ignore[index]


class TestMixedGroups:
    """Ownership is per handler; a foreign sibling is never disturbed."""

    def test_a_mixed_group_ends_with_one_aelfrice_handler(
        self, tmp_path: Path,
    ) -> None:
        hooks_path = tmp_path / "hooks.json"
        _write(hooks_path, {
            "UserPromptSubmit": [
                {"hooks": [_handler(WIN_HOOK), _handler(FOREIGN)]},
            ],
        })
        install_codex_hooks(hooks_path, windows=True)
        groups = _read(hooks_path)["UserPromptSubmit"]
        commands = [
            h["command"]
            for g in groups  # type: ignore[union-attr]
            for h in g["hooks"]
        ]
        assert commands.count(FOREIGN) == 1
        assert WIN_HOOK not in commands
        assert sum("aelf-hook" in c for c in commands) == 1

    def test_unsetup_strips_only_our_handler_from_a_mixed_group(
        self, tmp_path: Path,
    ) -> None:
        """The data-loss asymmetry, asserted directly.

        Group-level ownership left this group untouched on removal, so the
        aelfrice handler stayed wired forever after an uninstall.
        """
        hooks_path = tmp_path / "hooks.json"
        _write(hooks_path, {
            "UserPromptSubmit": [
                {"matcher": "*", "hooks": [_handler(WIN_HOOK), _handler(FOREIGN)]},
            ],
        })
        result = remove_codex_hooks(hooks_path, windows=True)

        assert result.changed is True
        groups = _read(hooks_path)["UserPromptSubmit"]
        assert len(groups) == 1
        group = groups[0]  # type: ignore[index]
        assert group["matcher"] == "*", "unrelated group keys must survive"
        assert group["hooks"] == [_handler(FOREIGN)]

    def test_unsetup_drops_a_group_that_becomes_empty(
        self, tmp_path: Path,
    ) -> None:
        hooks_path = tmp_path / "hooks.json"
        _write(hooks_path, {
            "UserPromptSubmit": [{"hooks": [_handler(WIN_HOOK)]}],
            "Stop": [{"hooks": [_handler(FOREIGN)]}],
        })
        remove_codex_hooks(hooks_path, windows=True)
        hooks = _read(hooks_path)
        assert "UserPromptSubmit" not in hooks, "emptied event must be dropped"
        assert hooks["Stop"] == [{"hooks": [_handler(FOREIGN)]}]

    def test_a_foreign_only_file_is_left_untouched(
        self, tmp_path: Path,
    ) -> None:
        hooks_path = tmp_path / "hooks.json"
        _write(hooks_path, {"Stop": [{"hooks": [_handler(FOREIGN)]}]})
        before = hooks_path.read_bytes()
        result = remove_codex_hooks(hooks_path, windows=True)
        assert result.changed is False
        assert hooks_path.read_bytes() == before

    def test_install_then_uninstall_round_trips_a_mixed_group(
        self, tmp_path: Path,
    ) -> None:
        """Setup and unsetup must agree on what "ours" means.

        AC: doctor and unsetup use exactly the same ownership primitive as
        setup. If they diverge, this leaves residue.
        """
        hooks_path = tmp_path / "hooks.json"
        _write(hooks_path, {
            "UserPromptSubmit": [{"hooks": [_handler(FOREIGN)]}],
        })
        before = json.loads(hooks_path.read_text(encoding="utf-8"))
        install_codex_hooks(hooks_path, windows=True)
        remove_codex_hooks(hooks_path, windows=True)
        assert json.loads(hooks_path.read_text(encoding="utf-8")) == before


class TestDoctor:
    def test_doctor_recognises_windows_launchers(self, tmp_path: Path) -> None:
        """AC: doctor reports the handlers and no missing events."""
        install_codex_hooks(tmp_path / "hooks.json", windows=True)
        report = doctor_codex(tmp_path, windows=True)
        assert report.owned_handler_count > 0
        assert report.missing_events == []

    def test_doctor_counts_a_handler_inside_a_mixed_group(
        self, tmp_path: Path,
    ) -> None:
        """Group-level counting made a mixed-group handler invisible."""
        hooks_path = tmp_path / "hooks.json"
        _write(hooks_path, {
            "UserPromptSubmit": [
                {"hooks": [_handler(WIN_HOOK), _handler(FOREIGN)]},
            ],
        })
        report = doctor_codex(tmp_path, windows=True)
        assert report.owned_handler_count == 1
        assert "UserPromptSubmit" not in report.missing_events

    def test_zero_recognised_handlers_is_no_longer_silent(
        self, tmp_path: Path,
    ) -> None:
        """The worst state used to emit an empty warning list.

        `owned_handler_count and missing_events` is falsy at zero handlers,
        so a valid hooks.json with nothing of ours in it produced no warning
        — precisely what the Windows reporter saw.
        """
        hooks_path = tmp_path / "hooks.json"
        _write(hooks_path, {
            "UserPromptSubmit": [{"hooks": [_handler(FOREIGN)]}],
        })
        report = doctor_codex(tmp_path, windows=True)
        assert report.owned_handler_count == 0
        assert report.missing_events == sorted(desired_codex_hooks())
        assert any("no aelfrice hook handlers" in w for w in report.warnings)

    def test_partial_coverage_still_warns_about_the_gap(
        self, tmp_path: Path,
    ) -> None:
        hooks_path = tmp_path / "hooks.json"
        _write(hooks_path, {
            "UserPromptSubmit": [{"hooks": [_handler(WIN_HOOK)]}],
        })
        report = doctor_codex(tmp_path, windows=True)
        assert any("coverage incomplete" in w for w in report.warnings)


@pytest.mark.parametrize("windows", [None, True])
def test_a_clean_install_is_idempotent(
    tmp_path: Path, windows: bool | None,
) -> None:
    """`None` is the host's own semantics; `True` forces the other branch.

    There is deliberately no `False` arm. On a Windows host the commands
    `desired_codex_hooks` resolves are backslash paths, and reading those
    under POSIX rules *correctly* fails to recognise them — so a `False` arm
    would assert idempotence against a deliberate mismatch and fail on
    `windows-smoke` only. That direction is covered, as a mismatch, by
    `test_the_same_file_still_duplicates_under_posix_rules`.
    """
    hooks_path = tmp_path / "hooks.json"
    install_codex_hooks(hooks_path, windows=windows)
    snapshot = hooks_path.read_bytes()
    assert install_codex_hooks(hooks_path, windows=windows).changed is False
    assert hooks_path.read_bytes() == snapshot
