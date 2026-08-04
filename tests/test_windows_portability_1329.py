"""#1329: aelfrice must import and lock on a host without `fcntl`.

The reported defect is not subtle — `session_ring` and `auto_install` both
did `import fcntl` at module scope, `cli` imports `auto_install` eagerly, so
on Windows every command died at import before argument parsing. `aelf
--help` and `aelf doctor` included.

Nobody on the team has a Windows host, so the arms here are built to run on
POSIX CI and still be load-bearing:

* the import arm masks `fcntl` out of a *fresh interpreter* rather than
  monkeypatching this one, so it reproduces the real failure mode (module
  scope, first import) instead of a simulation of it;
* the backend arms drive `file_lock` against injected fakes, so the Windows
  code path is executed and asserted on a machine that has no `msvcrt`.

What these cannot do is confirm that `msvcrt.locking` behaves as documented
on a real Windows host. That is what the `windows-latest` CI job added with
this change is for.
"""
from __future__ import annotations

import errno
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from aelfrice import file_lock

# Every module the CLI reaches at import time, plus the two that carried the
# unguarded import. Kept as a literal so a new module does not silently opt
# out of the guarantee — see `test_the_masked_import_list_covers_the_cli`.
_IMPORT_SMOKE_MODULES = (
    "aelfrice.cli",
    "aelfrice.auto_install",
    "aelfrice.session_ring",
    "aelfrice.file_lock",
    "aelfrice.hook",
)


@pytest.mark.timeout(120)
def test_every_cli_module_imports_without_fcntl() -> None:
    """A fresh interpreter with no `fcntl` must import the CLI.

    This is the regression arm for the actual field report. It runs in a
    subprocess because the failure is at *module scope on first import*:
    once this process has imported `aelfrice.cli` successfully, no amount
    of monkeypatching reproduces it.
    """
    script = textwrap.dedent(
        f"""
        import sys

        class Blocker:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "fcntl":
                    raise ImportError("no module named 'fcntl' (simulated)")
                return None

        # Evict any pre-imported copy, then refuse all future imports.
        sys.modules.pop("fcntl", None)
        sys.meta_path.insert(0, Blocker())

        try:
            import fcntl  # noqa: F401
        except ImportError:
            pass
        else:
            print("MASK-FAILED: fcntl still importable", file=sys.stderr)
            raise SystemExit(2)

        for name in {_IMPORT_SMOKE_MODULES!r}:
            __import__(name)

        from aelfrice.file_lock import HAVE_ADVISORY_LOCKS
        print("OK", HAVE_ADVISORY_LOCKS)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert proc.returncode == 0, (
        f"import failed without fcntl:\n{proc.stderr}"
    )
    assert proc.stdout.startswith("OK"), proc.stdout


def test_the_masked_import_list_covers_the_cli() -> None:
    """`cli` is what users invoke, so it must be in the smoke list.

    Without this, someone trimming the list to speed the test up could
    remove the one entry that reproduces the reported failure.
    """
    assert "aelfrice.cli" in _IMPORT_SMOKE_MODULES
    assert "aelfrice.auto_install" in _IMPORT_SMOKE_MODULES
    assert "aelfrice.session_ring" in _IMPORT_SMOKE_MODULES


# ---------------------------------------------------------------------------
# The Windows backend, exercised on POSIX via injection
# ---------------------------------------------------------------------------


class _FakeMsvcrt:
    """Enough of `msvcrt` to drive `file_lock`'s Windows path.

    `raise_errno` makes the next `locking()` call fail the way Windows
    fails: a plain `OSError`, which is *not* a `BlockingIOError`.
    """

    LK_LOCK = 0
    LK_NBLCK = 1
    LK_UNLCK = 2

    def __init__(self, raise_errno: int | None = None) -> None:
        self.raise_errno = raise_errno
        self.calls: list[tuple[int, int, int]] = []

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        self.calls.append((fd, mode, nbytes))
        if self.raise_errno is not None:
            raise OSError(self.raise_errno, "simulated")


@pytest.fixture
def windows_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Swap `file_lock` onto the Windows backend for one test."""

    def _install(fake: _FakeMsvcrt) -> _FakeMsvcrt:
        monkeypatch.setattr(file_lock, "_fcntl", None)
        monkeypatch.setattr(file_lock, "_msvcrt", fake)
        return fake

    return _install


@pytest.fixture
def lock_fd(tmp_path: Path) -> Any:
    fd = os.open(str(tmp_path / "x.lock"), os.O_CREAT | os.O_RDWR, 0o600)
    yield fd
    os.close(fd)


class TestWindowsBackend:
    def test_contention_is_reported_as_BlockingIOError(
        self, windows_backend: Any, lock_fd: int
    ) -> None:
        """The normalisation that keeps `auto_install` working.

        `msvcrt` raises a bare `OSError(EACCES)` where `fcntl` raises
        `BlockingIOError`. `auto_install.maybe_install_manifest` catches
        only `BlockingIOError` — unnormalised, the exception would escape
        the merge and surface as a crash at CLI entry, which is the exact
        class of failure this issue is about.
        """
        windows_backend(_FakeMsvcrt(raise_errno=errno.EACCES))
        with pytest.raises(BlockingIOError):
            file_lock.try_lock_exclusive(lock_fd)

    def test_a_non_contention_error_is_not_disguised_as_contention(
        self, windows_backend: Any, lock_fd: int
    ) -> None:
        """`session_ring._flock_until` distinguishes the two: contention
        means wait, anything else means this filesystem cannot lock and the
        caller should fall through unlocked. Collapsing them would turn an
        unsupported filesystem into an infinite poll."""
        windows_backend(_FakeMsvcrt(raise_errno=errno.EPERM))
        with pytest.raises(OSError) as caught:
            file_lock.try_lock_exclusive(lock_fd)
        assert not isinstance(caught.value, BlockingIOError)

    def test_the_file_position_is_restored(
        self, windows_backend: Any, lock_fd: int
    ) -> None:
        """`msvcrt.locking` is position-relative. An unbalanced seek would
        lock one region and unlock another — a permanently held lock, which
        is strictly worse than not locking."""
        windows_backend(_FakeMsvcrt())
        os.lseek(lock_fd, 7, os.SEEK_SET)
        file_lock.lock_exclusive(lock_fd)
        assert os.lseek(lock_fd, 0, os.SEEK_CUR) == 7
        file_lock.unlock(lock_fd)
        assert os.lseek(lock_fd, 0, os.SEEK_CUR) == 7

    def test_lock_and_unlock_act_on_the_same_region(
        self, windows_backend: Any, lock_fd: int
    ) -> None:
        fake = windows_backend(_FakeMsvcrt())
        file_lock.lock_exclusive(lock_fd)
        file_lock.unlock(lock_fd)
        assert [c[2] for c in fake.calls] == [1, 1], "region size drifted"
        assert [c[1] for c in fake.calls] == [
            _FakeMsvcrt.LK_LOCK,
            _FakeMsvcrt.LK_UNLCK,
        ]


class TestNoBackendFallback:
    """No `fcntl` and no `msvcrt`: degrade to no-op, never raise."""

    @pytest.fixture(autouse=True)
    def _no_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(file_lock, "_fcntl", None)
        monkeypatch.setattr(file_lock, "_msvcrt", None)

    def test_all_three_operations_are_silent_no_ops(
        self, lock_fd: int
    ) -> None:
        """Raising here would convert "this host cannot lock" into a crash
        at CLI entry — `auto_install` catches only `BlockingIOError`."""
        file_lock.lock_exclusive(lock_fd)
        file_lock.try_lock_exclusive(lock_fd)
        file_lock.unlock(lock_fd)

    def test_the_degradation_is_reported_not_hidden(self) -> None:
        """`HAVE_ADVISORY_LOCKS` is what lets `doctor` and the docs say
        that concurrent processes are not serialised on this host. A silent
        no-op lock reads as serialisation that is not there."""
        assert file_lock._have_advisory_locks() is False


def test_posix_hosts_still_report_real_locks() -> None:
    """Guards the fixtures above: if the no-backend path leaked into the
    module state, this fails and the suite stops trusting the others."""
    if sys.platform == "win32":  # pragma: no cover - not our CI default
        pytest.skip("POSIX-only assertion")
    assert file_lock.HAVE_ADVISORY_LOCKS is True
