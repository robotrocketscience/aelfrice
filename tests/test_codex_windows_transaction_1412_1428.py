"""#1412 ownership through the #1428 hooks.json transaction.

The two changes met in one file. #1412 added a `windows` platform seam and
threaded it through ownership; #1428 turned `install_codex_hooks` /
`remove_codex_hooks` into one-line delegations into
`_commit_hooks_transaction`, so the ownership decision now lives in
`_plan_install` / `_plan_remove` and is re-executed on every commit attempt.

`tests/test_codex_windows_ownership_1412.py` asserts the outcome of the
public functions. It does not reach the transaction's own machinery — the
lock, the fingerprint check, or the re-plan on a losing fingerprint — so it
is not evidence that the seam survives them. This file covers that seam, and
it matters because ownership drives a deletion path: a `windows` that fails
to reach the re-planned closure silently narrows or widens what is removed,
and POSIX CI cannot see the Windows branch any other way.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice import host_codex
from aelfrice.host_codex import (
    _plan_install,
    _plan_remove,
    install_codex_hooks,
    remove_codex_hooks,
)

WIN_HOOK = r"C:\Users\dev\.venv\Scripts\aelf-hook.EXE"
POSIX_HOOK = "/nonexistent/bin/aelf-hook"
FOREIGN = r"C:\Tools\foreign-hook.EXE"


def _h(command: str) -> dict[str, object]:
    return {"type": "command", "command": command}


def _doc(hooks: dict[str, object]) -> str:
    return json.dumps({"hooks": hooks}, indent=2) + "\n"


def _commands(path: Path, event: str) -> list[str]:
    groups = json.loads(path.read_text(encoding="utf-8"))["hooks"].get(event, [])
    return [h["command"] for g in groups for h in g["hooks"]]


class TestThePlanFunctionsCarryTheSeam:
    """The merge point: the seam has to reach the pure planners."""

    def test_plan_install_claims_an_exe_launcher_under_windows(
        self, tmp_path: Path,
    ) -> None:
        text = _doc({"UserPromptSubmit": [{"hooks": [_h(WIN_HOOK)]}]})
        serialized, _ = _plan_install(
            text, "user", False, tmp_path / "hooks.json", windows=True,
        )
        assert serialized is not None
        groups = json.loads(serialized)["hooks"]["UserPromptSubmit"]
        commands = [h["command"] for g in groups for h in g["hooks"]]
        assert WIN_HOOK not in commands, commands
        assert len(groups) == 1, groups

    def test_plan_install_does_not_claim_it_under_posix(
        self, tmp_path: Path,
    ) -> None:
        """The distinguishing arm. Widening ownership is the unsafe way."""
        text = _doc({"UserPromptSubmit": [{"hooks": [_h(WIN_HOOK)]}]})
        serialized, _ = _plan_install(
            text, "user", False, tmp_path / "hooks.json", windows=False,
        )
        assert serialized is not None
        groups = json.loads(serialized)["hooks"]["UserPromptSubmit"]
        commands = [h["command"] for g in groups for h in g["hooks"]]
        assert WIN_HOOK in commands, commands
        assert len(groups) == 2, groups

    def test_plan_remove_strips_only_the_exe_handler_under_windows(
        self, tmp_path: Path,
    ) -> None:
        text = _doc({"Stop": [
            {"matcher": "MIX", "hooks": [_h(WIN_HOOK), _h(FOREIGN)]},
        ]})
        serialized, result = _plan_remove(
            text, tmp_path / "hooks.json", windows=True,
        )
        assert result.changed is True
        assert serialized is not None
        groups = json.loads(serialized)["hooks"]["Stop"]
        assert groups == [{"matcher": "MIX", "hooks": [_h(FOREIGN)]}]

    def test_plan_remove_leaves_it_alone_under_posix(
        self, tmp_path: Path,
    ) -> None:
        text = _doc({"Stop": [
            {"matcher": "MIX", "hooks": [_h(WIN_HOOK), _h(FOREIGN)]},
        ]})
        serialized, result = _plan_remove(
            text, tmp_path / "hooks.json", windows=False,
        )
        assert result.changed is False
        assert serialized is None


class TestThroughTheCommitPath:
    """The transaction itself: lock, fingerprint, atomic replace, re-plan."""

    def test_an_exe_launcher_is_recognised_through_the_commit(
        self, tmp_path: Path,
    ) -> None:
        hooks_path = tmp_path / "hooks.json"
        hooks_path.write_text(
            _doc({"UserPromptSubmit": [{"hooks": [_h(WIN_HOOK)]}]}), "utf-8",
        )
        install_codex_hooks(hooks_path, windows=True)

        assert WIN_HOOK not in _commands(hooks_path, "UserPromptSubmit")
        # The commit went through the transaction, not around it.
        assert (tmp_path / "hooks.json.lock").is_file()
        assert not list(tmp_path.glob("hooks.json.*.tmp")), "temp file leaked"

    def test_the_seam_survives_a_replan_after_a_losing_fingerprint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The arm the old ownership test cannot reach.

        `_commit_under_lock` re-invokes the plan closure when the file
        changed under it. `windows` is captured by that closure, so a merge
        that threaded the seam into the *first* call only — or that let the
        retry rebuild a default — reads the platform differently on the
        second attempt. Here the retry is forced: a foreign writer replaces
        the file between our read and our replace, so attempt one loses its
        fingerprint and attempt two plans against the newer document.
        """
        hooks_path = tmp_path / "hooks.json"
        hooks_path.write_text(
            _doc({"UserPromptSubmit": [{"hooks": [_h(WIN_HOOK)]}]}), "utf-8",
        )

        reads: list[int] = []
        real_read = host_codex._read_hooks_snapshot

        def racing_read(path: Path) -> tuple[str | None, str]:
            snapshot = real_read(path)
            reads.append(1)
            if len(reads) == 1:
                # Someone else commits between our read and our replace.
                path.write_text(
                    _doc({
                        "UserPromptSubmit": [{"hooks": [_h(WIN_HOOK)]}],
                        "Stop": [{"hooks": [_h(FOREIGN)]}],
                    }),
                    encoding="utf-8",
                )
            return snapshot

        monkeypatch.setattr(host_codex, "_read_hooks_snapshot", racing_read)
        result = install_codex_hooks(hooks_path, windows=True)

        assert result.error is None
        assert len(reads) == 2, "the losing fingerprint must force one retry"
        # Ownership still held on the retry: replaced, not appended...
        assert WIN_HOOK not in _commands(hooks_path, "UserPromptSubmit")
        # ...and the foreign writer's entry was merged, not clobbered.
        assert FOREIGN in _commands(hooks_path, "Stop")

    @pytest.mark.parametrize("windows", [True, False])
    def test_the_deletion_matrix_is_pinned_on_both_platforms(
        self, tmp_path: Path, windows: bool,
    ) -> None:
        """One file, three handlers, both platform values, exact survivors.

        POSIX must claim the POSIX launcher and *not* the `.EXE` one;
        Windows must claim both. The foreign handler survives either way.
        This is the acceptance for the merge: the numbers here are the same
        numbers `remove_codex_hooks` produced before the transaction existed.
        """
        hooks_path = tmp_path / "hooks.json"
        hooks_path.write_text(
            _doc({"Stop": [{"matcher": "MIX", "hooks": [
                _h(POSIX_HOOK), _h(WIN_HOOK), _h(FOREIGN),
            ]}]}),
            encoding="utf-8",
        )
        result = remove_codex_hooks(hooks_path, windows=windows)

        assert result.changed is True
        survivors = _commands(hooks_path, "Stop")
        if windows:
            assert survivors == [FOREIGN]
        else:
            assert survivors == [WIN_HOOK, FOREIGN]
