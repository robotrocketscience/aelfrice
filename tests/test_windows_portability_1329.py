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
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import pytest

from aelfrice import file_lock
from aelfrice.stream_encoding import ensure_utf8_streams
from aelfrice.claude_memory import (
    derive_memory_dir,
    encode_project_path,
)

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


# ---------------------------------------------------------------------------
# The claude-memory path encoder
# ---------------------------------------------------------------------------


class TestProjectPathEncoding:
    """`derive_memory_dir` built `-C:\\Dev\\example\\proj` on Windows.

    The old rule stripped a leading slash and replaced `/`, then prepended a
    hardcoded `-`. On a Windows path none of the three steps did what they
    were written for, and the result was a directory that cannot exist — so
    both claude-memory commands and the #985 mirror silently found nothing.

    These drive the pure string helper, because `Path.resolve()` cannot
    construct a foreign-flavour absolute path from POSIX CI.
    """

    @pytest.mark.parametrize(
        ("abs_path", "expected"),
        [
            # POSIX: unchanged from the pre-#1329 encoding. Regression arm —
            # the fix must not move the directory every existing user's
            # memories already live in.
            ("/Users/alice/projects/myapp", "-Users-alice-projects-myapp"),
            ("/a", "-a"),
            # Windows: the reported case. The leading dash is not a prefix,
            # it falls out of the leading separator; a drive letter has none,
            # so `C:` becomes `C--` and the name starts with the drive.
            (r"C:\Dev\example\proj", "C--Dev-example-proj"),
            (r"D:\x", "D--x"),
            # UNC: both leading separators are replaced, like any others.
            (r"\\server\share\proj", "--server-share-proj"),
            # A dotted component. The doubled dash is the separator and
            # the dot each mapping to one -- not a special leading-dot
            # rule. Taken from a real host directory rather than
            # derived: `~/.claude/projects` held 97 entries and none
            # contained a literal `.`.
            (
                "/Users/alice/projects/app/.claude/worktrees/7",
                "-Users-alice-projects-app--claude-worktrees-7",
            ),
            ("/Users/alice/my.app", "-Users-alice-my-app"),
            (r"C:\Dev\my.app", "C--Dev-my-app"),
        ],
    )
    def test_the_encoding_is_one_substitution_over_both_flavours(
        self, abs_path: str, expected: str
    ) -> None:
        assert encode_project_path(abs_path) == expected

    def test_no_dot_survives_into_a_directory_name(self) -> None:
        """The POSIX half of the same defect the Windows half names.

        A surviving dot is a legal filename character, so it does not fail
        loudly the way a surviving colon does -- the directory is simply
        not the one the host tool made, and every lookup finds nothing.
        Any path with a dotted component is affected, which includes every
        project under a `.claude/` or `.config/` directory.
        """
        encoded = encode_project_path("/Users/alice/.claude/worktrees/7")
        assert "." not in encoded
        assert encoded == "-Users-alice--claude-worktrees-7"

    def test_no_colon_survives_into_a_directory_name(self) -> None:
        """The specific defect. `:` is not a legal filename character on
        Windows, so a surviving colon is not merely a wrong directory — it
        is an uncreatable one."""
        assert ":" not in encode_project_path(r"C:\Dev\example\proj")

    def test_no_backslash_survives_into_a_directory_name(self) -> None:
        """A surviving backslash would be read as a path separator and
        silently nest the memory dir several levels deeper."""
        assert "\\" not in encode_project_path(r"C:\Dev\example\proj")

    def test_a_windows_path_encodes_to_exactly_one_path_segment(self) -> None:
        """The encoded string is joined as a single directory name. If any
        separator survived, `PurePath` would see multiple segments and the
        memory dir would land somewhere else entirely."""
        encoded = encode_project_path(r"C:\Dev\example\proj")
        assert len(PureWindowsPath(encoded).parts) == 1
        assert len(PurePosixPath(encoded).parts) == 1

    def test_derive_memory_dir_still_ends_at_a_memory_directory(
        self, tmp_path: Path
    ) -> None:
        """The join is unchanged; only the encoding moved into a helper."""
        got = derive_memory_dir(tmp_path)
        assert got.name == "memory"
        assert got.parent.parent.name == "projects"
        assert got.parent.name == encode_project_path(str(tmp_path.resolve()))


# ---------------------------------------------------------------------------
# Console encoding
# ---------------------------------------------------------------------------


class _FakeStream:
    """A stream that encodes like a legacy Windows console.

    `reconfigure` records the request the way `TextIOWrapper` would honour
    it, so the test asserts on what the code *asked for* rather than on
    what a POSIX terminal happens to do.
    """

    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self.reconfigured: tuple[str, str] | None = None

    def reconfigure(self, *, encoding: str, errors: str) -> None:
        self.reconfigured = (encoding, errors)
        self.encoding = encoding


class _StreamWithoutReconfigure:
    """A `StringIO`-shaped stream: no `reconfigure`, no charmap codec."""

    def __init__(self, encoding: str) -> None:
        self.encoding = encoding


class TestConsoleEncoding:
    """`aelf --help` crashed on Windows *after* the fcntl fix.

    argparse ran, then `print_help` hit `cp1252.encode` and raised
    `UnicodeEncodeError` on an em dash. This surfaced only because the
    windows-latest job exists — the POSIX suite cannot see it, since POSIX
    streams are already UTF-8.
    """

    def test_a_cp1252_stream_is_reconfigured(self) -> None:
        stream = _FakeStream("cp1252")
        ensure_utf8_streams((stream,))  # type: ignore[arg-type]
        assert stream.reconfigured == ("utf-8", "replace")

    def test_a_utf8_stream_is_left_alone(self) -> None:
        """No-op on POSIX. Reconfiguring an already-UTF-8 stream would be
        a needless side effect at every process entry."""
        for spelling in ("utf-8", "UTF-8", "utf8"):
            stream = _FakeStream(spelling)
            ensure_utf8_streams((stream,))  # type: ignore[arg-type]
            assert stream.reconfigured is None, spelling

    def test_a_stream_that_cannot_reconfigure_is_skipped(self) -> None:
        """`out` is a StringIO under test and a pipe wrapper in places.
        Neither goes through a charmap codec, and neither should raise."""
        stream = _StreamWithoutReconfigure("cp1252")
        assert not hasattr(stream, "reconfigure")
        ensure_utf8_streams((stream,))  # type: ignore[arg-type]

    def test_a_raising_reconfigure_is_swallowed(self) -> None:
        """A detached or closed stream must not turn into an exception at
        process entry — that is the failure mode being fixed, not a new
        place to introduce it."""

        class _Detached(_FakeStream):
            def reconfigure(self, *, encoding: str, errors: str) -> None:
                raise ValueError("underlying buffer has been detached")

        ensure_utf8_streams((_Detached("cp1252"),))  # type: ignore[arg-type]

    def test_help_text_contains_characters_cp1252_cannot_encode(self) -> None:
        """Pins *why* this is needed. If the help text were pure ASCII the
        reconfigure would be dead code; it is not, and this fails loudly if
        someone concludes otherwise and removes the call."""
        from aelfrice.cli import build_parser

        help_text = build_parser().format_help()
        with pytest.raises(UnicodeEncodeError):
            help_text.encode("cp1252")
