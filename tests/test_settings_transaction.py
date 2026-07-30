"""settings.json mutations are serialised and batched (#1161).

`aelf setup` performed roughly ten independent read-modify-write cycles
on `settings.json` while holding no lock of any kind. `auto_install`
locks `~/.aelfrice/.auto-install.lock`, which serialises auto-install
against *itself* and says nothing about `aelf setup`, `aelf unsetup`, or
`aelf doctor --fix`, none of which took any lock. `_atomic_write` is
atomic per write but is not a compare-and-swap, so a concurrent writer's
whole-file state was silently discarded.

Reproduced before the fix with two threads that load the same document
and write in sequence: the first writer's key is absent afterwards.

One correction to the filed report, which asked for the ten cycles to be
collapsed into a single load-once/write-once pass. Collapsing *alone*
would have made things worse. Each cycle re-reads the file today, so a
host write landing between two installers survives (measured: a
permission grant injected mid-install is still present afterwards).
A single pass widens that window from ten microsecond-scale gaps to the
whole duration of `aelf setup`. What makes batching safe is the
fingerprint check at commit — a foreign write is detected and the
transaction aborts rather than overwriting it.
"""
from __future__ import annotations

import argparse
import io
import json
import threading
import time
from pathlib import Path

import pytest

from aelfrice import setup as S
from aelfrice.session_ring import FileLockTimeout, exclusive_file_lock
from aelfrice.setup import (
    SettingsChangedDuringTransaction,
    settings_transaction,
)


@pytest.fixture
def settings(tmp_path: Path) -> Path:
    """A settings.json holding one user-owned key we must never lose."""
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"permissions": {"allow": ["Bash(ls)"]}}, indent=2) + "\n"
    )
    return path


@pytest.fixture
def count_writes(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every real settings write; the list length is the count."""
    seen: list[Path] = []
    real = S._write_settings_file

    def counting(path: Path, data: dict[str, object]) -> None:
        seen.append(path)
        real(path, data)

    monkeypatch.setattr(S, "_write_settings_file", counting)
    return seen


def _hook_entry_count(path: Path) -> int:
    data = json.loads(path.read_text())
    return sum(
        len(entry.get("hooks", []))
        for entries in (data.get("hooks") or {}).values()
        for entry in entries
    )


# --- Serialisation ------------------------------------------------------


def test_a_contended_transaction_times_out_rather_than_racing(
    settings: Path,
) -> None:
    """The defect was that a second writer proceeded regardless.

    Timing-free assertion of mutual exclusion: while one transaction is
    open, a second cannot acquire. Pre-#1161 there was no lock at all, so
    the second writer sailed through and clobbered the first.
    """
    holding = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with settings_transaction(settings):
            holding.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert holding.wait(timeout=5)
    try:
        with pytest.raises(FileLockTimeout):
            with settings_transaction(settings, timeout=0.05):
                pass
    finally:
        release.set()
        thread.join(timeout=5)


def test_two_serialised_writers_both_survive(settings: Path) -> None:
    """The end-to-end property the lock buys: no lost update.

    Both threads load, mutate, and write. Pre-#1161 the loser's key was
    simply gone from the file.
    """
    start = threading.Barrier(2)
    errors: list[str] = []

    def writer(key: str, delay: float) -> None:
        try:
            start.wait(timeout=5)
            time.sleep(delay)
            with settings_transaction(settings, timeout=5):
                data = S._load_settings(settings)
                time.sleep(0.05)  # widen the window the lock must cover
                data[key] = f"by-{key}"
                S._atomic_write(settings, data)
        except Exception as exc:  # noqa: BLE001 - surfaced via assert
            errors.append(repr(exc))

    threads = [
        threading.Thread(target=writer, args=("A", 0.0)),
        threading.Thread(target=writer, args=("B", 0.01)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    final = json.loads(settings.read_text())
    assert final.get("A") == "by-A"
    assert final.get("B") == "by-B"
    assert final["permissions"]["allow"] == ["Bash(ls)"]


def test_the_transaction_slot_is_per_thread(settings: Path) -> None:
    """A module-global slot would leak one thread's transaction to another.

    Caught during development: with a plain global, an unrelated thread
    saw the holder's transaction and tripped the nesting guard. A process
    can legitimately run two of these (a threaded host, the MCP server).
    """
    holding = threading.Event()
    release = threading.Event()
    observed: list[object] = []

    def holder() -> None:
        with settings_transaction(settings):
            holding.set()
            release.wait(timeout=5)

    thread = threading.Thread(target=holder, daemon=True)
    thread.start()
    assert holding.wait(timeout=5)
    try:
        observed.append(S._active_transaction())
    finally:
        release.set()
        thread.join(timeout=5)
    assert observed == [None], "another thread's transaction was visible"


# --- Batching -----------------------------------------------------------


def test_many_installers_produce_one_write(
    settings: Path, count_writes: list[Path]
) -> None:
    installers = [
        S.install_user_prompt_submit_hook,
        S.install_session_start_hook,
        S.install_stop_hook,
        S.install_commit_ingest_hook,
        S.install_search_tool_hook,
        S.install_search_tool_bash_hook,
    ]
    with settings_transaction(settings):
        for i, install in enumerate(installers):
            install(settings, command=f"/usr/local/bin/aelf-h{i}", timeout=15)
    assert len(count_writes) == 1
    assert _hook_entry_count(settings) == len(installers)


def test_a_transaction_that_changes_nothing_writes_nothing(
    settings: Path, count_writes: list[Path]
) -> None:
    """The "already present, skip the write" path must stay a no-op.

    `_install_or_replace_entry` returning unmutated means the installer
    never calls the writer, so the transaction must not be marked dirty
    and must not touch the disk on commit.
    """
    with settings_transaction(settings):
        S.install_stop_hook(
            settings, command="/usr/local/bin/aelf-stop-hook", timeout=15
        )
    count_writes.clear()
    with settings_transaction(settings):
        S.install_stop_hook(
            settings, command="/usr/local/bin/aelf-stop-hook", timeout=15
        )
    assert count_writes == []


def test_mutations_accumulate_in_one_document(settings: Path) -> None:
    """Successive `_load_settings` calls inside a transaction share state."""
    with settings_transaction(settings):
        first = S._load_settings(settings)
        first["marker"] = 1
        S._atomic_write(settings, first)
        second = S._load_settings(settings)
        assert second is first
        assert second["marker"] == 1


def test_a_write_to_another_path_bypasses_the_transaction(
    tmp_path: Path, settings: Path
) -> None:
    """Only the transaction's own path is buffered.

    A project-scope settings file written while a user-scope transaction
    is open must land immediately rather than be silently swallowed.
    """
    other = tmp_path / "project-settings.json"
    with settings_transaction(settings):
        S._atomic_write(other, {"other": True})
        assert json.loads(other.read_text()) == {"other": True}


# --- Safety at commit ---------------------------------------------------


def test_a_foreign_write_aborts_the_commit(settings: Path) -> None:
    """The host owns this file too and cannot be made to take our lock.

    Committing over its write would discard it, which is the failure the
    filed report described. Detect and abort instead.
    """
    with pytest.raises(SettingsChangedDuringTransaction):
        with settings_transaction(settings):
            data = S._load_settings(settings)
            data["ours"] = True
            S._atomic_write(settings, data)
            # The host grants a permission, taking no aelfrice lock.
            host = json.loads(settings.read_text())
            host["permissions"]["allow"].append("Bash(git push:*)")
            S._write_settings_file(settings, host)

    final = json.loads(settings.read_text())
    assert final["permissions"]["allow"] == ["Bash(ls)", "Bash(git push:*)"]
    assert "ours" not in final, "our commit overwrote the foreign write"


def test_an_exception_in_the_block_discards_the_buffer(
    settings: Path, count_writes: list[Path]
) -> None:
    """A half-applied hook set must never reach the disk."""
    with pytest.raises(ValueError, match="boom"):
        with settings_transaction(settings):
            data = S._load_settings(settings)
            data["half"] = True
            S._atomic_write(settings, data)
            raise ValueError("boom")
    assert count_writes == []
    assert "half" not in json.loads(settings.read_text())


def test_the_transaction_slot_is_cleared_after_an_exception(
    settings: Path,
) -> None:
    """Otherwise every later mutation in the process buffers into a dead
    transaction and is silently dropped."""
    with pytest.raises(ValueError):
        with settings_transaction(settings):
            raise ValueError("boom")
    assert S._active_transaction() is None
    # And a plain write still reaches the disk.
    S._atomic_write(settings, {"after": True})
    assert json.loads(settings.read_text()) == {"after": True}


def test_nesting_is_refused(settings: Path) -> None:
    with pytest.raises(RuntimeError, match="does not nest"):
        with settings_transaction(settings):
            with settings_transaction(settings):
                pass


def test_a_missing_settings_file_commits_cleanly(tmp_path: Path) -> None:
    """Fingerprint of a nonexistent file is stable, so first install works."""
    path = tmp_path / "nested" / "settings.json"
    with settings_transaction(path):
        S.install_stop_hook(
            path, command="/usr/local/bin/aelf-stop-hook", timeout=15
        )
    assert _hook_entry_count(path) == 1


# --- Lock primitive ----------------------------------------------------


def test_exclusive_file_lock_still_blocks_by_default(tmp_path: Path) -> None:
    """`timeout=None` is the pre-#1161 contract the ring/telemetry use."""
    target = tmp_path / "ring.json"
    with exclusive_file_lock(target):
        pass
    with exclusive_file_lock(target, timeout=1.0):
        pass


def test_flock_until_tries_once_even_with_a_zero_timeout(
    tmp_path: Path,
) -> None:
    """A non-positive timeout must degrade to one attempt, not to failure."""
    target = tmp_path / "zero.json"
    with exclusive_file_lock(target, timeout=0.0):
        pass


def test_lock_timeout_is_not_the_unsupported_filesystem_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An OSError that is not contention keeps the unlocked fall-through.

    Advisory locking is unavailable on some filesystems; a writer there
    must still write rather than raise.
    """
    import errno as _errno
    import fcntl as _fcntl

    def unsupported(fd: int, op: int) -> None:
        raise OSError(_errno.ENOLCK, "no locks available")

    monkeypatch.setattr(_fcntl, "flock", unsupported)
    with exclusive_file_lock(tmp_path / "nolock.json", timeout=0.5):
        pass  # reached without raising


# --- The four call sites ------------------------------------------------


def _setup_args(settings_path: Path) -> argparse.Namespace:
    from aelfrice.cli import build_parser

    return build_parser(show_advanced=True).parse_args(
        ["setup", "--settings-path", str(settings_path)]
    )


def test_cmd_setup_writes_settings_once(
    settings: Path, count_writes: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from aelfrice import cli as C

    args = _setup_args(settings)
    # Redirect the slash-command bundle into the tmp tree rather than the
    # developer's real ~/.claude/commands/aelf/.
    args.slash_commands_dir = str(settings.parent / "commands")
    buf = io.StringIO()
    rc = C._cmd_setup_locked(args, buf)
    assert rc == 0
    settings_only = [p for p in count_writes if p == settings]
    assert len(settings_only) == 1, f"{len(settings_only)} writes"
    assert _hook_entry_count(settings) > 1
    assert json.loads(settings.read_text())["permissions"]["allow"] == [
        "Bash(ls)"
    ]


@pytest.mark.parametrize("command", ["setup", "unsetup"])
def test_a_contended_command_reports_and_exits_nonzero(
    settings: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """A traceback is the wrong answer to "someone else holds the lock"."""
    from aelfrice import cli as C

    def contended(path: Path, **kwargs: object) -> object:
        raise FileLockTimeout(f"could not acquire {path}.lock within 10.0s")

    monkeypatch.setattr(C, "settings_transaction", contended)
    parser = C.build_parser(show_advanced=True)
    args = parser.parse_args([command, "--settings-path", str(settings)])
    buf = io.StringIO()
    handler = C._cmd_setup if command == "setup" else C._cmd_unsetup
    assert handler(args, buf) == 1
    out = buf.getvalue()
    assert "aborted" in out
    assert "re-run" in out


def test_doctor_fix_skips_a_contended_scope(
    settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aelfrice import cli as C

    def contended(path: Path, **kwargs: object) -> object:
        raise FileLockTimeout("held by a sibling")

    monkeypatch.setattr(C, "settings_transaction", contended)
    report = argparse.Namespace(scopes_scanned=[("user", settings)])
    buf = io.StringIO()
    C._cmd_doctor_fix_hooks(
        argparse.Namespace(dry_run=False), report, buf
    )
    assert "skipped" in buf.getvalue()


def test_doctor_dry_run_needs_no_lock(
    settings: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run writes nothing, so a sibling's lock must not block it."""
    from aelfrice import cli as C

    def fail(path: Path, **kwargs: object) -> object:
        raise AssertionError("dry run must not open a transaction")

    monkeypatch.setattr(C, "settings_transaction", fail)
    report = argparse.Namespace(scopes_scanned=[("user", settings)])
    buf = io.StringIO()
    C._cmd_doctor_fix_hooks(argparse.Namespace(dry_run=True), report, buf)
    assert "--fix:" in buf.getvalue()


def test_the_prune_joins_an_open_transaction(settings: Path) -> None:
    """The prune used to write around whatever transaction was open.

    Its own atomic write would change the file under `aelf setup`'s
    transaction and trip the fingerprint check at commit, so routing it
    through `setup.write_settings` is load-bearing rather than cosmetic.
    """
    from aelfrice.doctor import prune_broken_aelf_hooks

    data = json.loads(settings.read_text())
    data["hooks"] = {
        "Stop": [
            {
                "hooks": [
                    {
                        "type": "command",
                        "command": "/nonexistent/bin/aelf-stop-hook",
                    }
                ]
            }
        ]
    }
    settings.write_text(json.dumps(data, indent=2) + "\n")

    with settings_transaction(settings):
        result = prune_broken_aelf_hooks(settings)
        assert result.total_removed == 1
        S.install_stop_hook(
            settings, command="/usr/local/bin/aelf-stop-hook", timeout=15
        )
    # Committed once, with both the prune and the install applied.
    final = json.loads(settings.read_text())
    commands = [
        inner["command"]
        for entries in final["hooks"].values()
        for entry in entries
        for inner in entry["hooks"]
    ]
    assert commands == ["/usr/local/bin/aelf-stop-hook"]


def test_auto_install_merge_writes_settings_once(
    tmp_path: Path, count_writes: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The merge installs ten hooks; it used to write ten times."""
    from aelfrice import auto_install as A

    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(A, "read_opt_outs", lambda *a, **k: set())
    result = A._do_merge(
        prev_version="0.0.0",
        installed_version="9.9.9",
        scope="user",
        settings_path=settings_path,
        stamp_path=tmp_path / "stamp",
        opt_out_path=tmp_path / "opt-out.json",
        timeout=None,
    )
    assert result.ran
    settings_only = [p for p in count_writes if p == settings_path]
    assert len(settings_only) == 1, f"{len(settings_only)} writes"
    assert _hook_entry_count(settings_path) >= 10
