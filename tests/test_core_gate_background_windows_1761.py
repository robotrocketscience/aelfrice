"""Background labeling on Windows, and the boundaries it shares with POSIX (#1761).

#1741 turned the background labeler off on Windows until three things were
checked there: that the detached child opens no console window, that an npm
`claude.cmd` shim receives the `--settings` argument intact, and that the
reply decodes. The windows-smoke workflow runs this file on `windows-latest`
with no `claude` login and no network, so every test here runs a stub in
place of the CLI: a `claude.cmd` on Windows, a shell script elsewhere, both
handing their arguments to a small Python program that records them and
replies with one label. Each test names the mutation that kills it. Every
store is under `tmp_path`, and the belief text is synthetic.
"""
from __future__ import annotations

import io
import json
import ntpath
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest

import aelfrice.core_gate_background as bg
from aelfrice.core_gate import CLASSIFIER_VERSION
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
    CoreGateBatchItem,
)
from aelfrice.store import MemoryStore

pytestmark = pytest.mark.timeout(60)

# Captured at import, before the autouse conftest fixture replaces it.
_REAL_FIND_CLAUDE = bg.find_claude

SESSION = "widgetsession-win"
CLAUDE = "/opt/widget/bin/claude"
WIN32 = sys.platform == "win32"

# Windows `CreateProcess` flags, by value (processthreadsapi.h). The module
# resolves them through `getattr` on `subprocess`, which has them only on
# Windows, so these are what the test can compare against on any host.
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

# The stub CLI: records its argv and the `--settings` file beside itself,
# then replies with one label and a key the ANSI code page can't decode.
# `東京` is e6 9d b1 e4 ba ac in UTF-8; 0x9d is undefined in cp1252
# and every byte is outside ASCII, so a locale decode raises on either.
_STUB_SOURCE = '''\
import json, sys
from pathlib import Path

here = Path(__file__).resolve().parent
argv = sys.argv[1:]
(here / "argv.json").write_text(json.dumps(argv), encoding="utf-8")
if "--settings" in argv:
    settings = Path(argv[argv.index("--settings") + 1])
    (here / "settings-copy.json").write_bytes(settings.read_bytes())
(here / "prompt.txt").write_bytes(sys.stdin.buffer.read())
reply = {
    "type": "result",
    "result": "```json\\n[{\\"index\\": 0, \\"label\\": \\"A\\"}]\\n```",
    "total_cost_usd": 0.01,
    "note": "\\u6771\\u4eac",
}
sys.stdout.buffer.write(json.dumps(reply, ensure_ascii=False).encode("utf-8"))
'''

# Forces a non-UTF-8 text-mode decode in a child interpreter: UTF-8 mode off,
# the PEP 538 coercion of C to C.UTF-8 off, and the C locale selected. On
# Windows the first line is what matters; the locale encoding there is the
# ANSI code page whatever LC_ALL says.
_LOCALE_ENV = {
    "PYTHONUTF8": "0",
    "PYTHONCOERCECLOCALE": "0",
    "LC_ALL": "C",
    "LANG": "C",
}


def _write_stub(folder: Path) -> str:
    """A `claude` the labeler can execute from `folder`; returns its path."""
    program = folder / "stub_claude.py"
    program.write_text(_STUB_SOURCE, encoding="utf-8")
    if WIN32:
        shim = folder / "claude.cmd"
        shim.write_text(
            f'@echo off\r\n"{sys.executable}" "{program}" %*\r\n', encoding="utf-8",
        )
        # npm's other two files beside the shim, which `find_claude` must
        # pass over: an extensionless `sh` script and a PowerShell script.
        (folder / "claude").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        (folder / "claude.ps1").write_text("exit 1\r\n", encoding="utf-8")
    else:
        shim = folder / "claude"
        shim.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{program}" "$@"\n', encoding="utf-8",
        )
        shim.chmod(0o755)
    return str(shim)


def _belief(bid: str) -> Belief:
    return Belief(
        id=bid, content=f"The widgetwin {bid} setting is stable.",
        content_hash=f"hash-{bid}", alpha=3.0, beta=1.0, type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE, locked_at=None, created_at="2026-10-01T00:00:00Z",
        last_retrieved_at=None, session_id=SESSION, origin=ORIGIN_USER_TRANSCRIPT,
    )


def _open_batch(db: Path) -> str:
    """A store with one belief and one open batch over it; returns the batch id."""
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_belief("w1"))
        return store.create_core_gate_batch(
            [CoreGateBatchItem(index=0, belief_id="w1", content_hash="hash-w1")],
            classifier_version=str(CLASSIFIER_VERSION), origin="session_end",
            session_id=SESSION, created_at="2026-10-01T00:00:00Z",
        )
    finally:
        store.close()


def _labels(db: Path) -> dict[str, str]:
    conn = sqlite3.connect(str(db))
    try:
        return dict(conn.execute("SELECT content_hash, label FROM core_gate_labels").fetchall())
    finally:
        conn.close()


# --- the decode boundary --------------------------------------------------------


# Asks the codec directly, not `subprocess`: a failed text-mode decode raises
# on POSIX, but on Windows it kills the reader thread and `run` returns
# `stdout=None`, so a probe through `subprocess` can't tell the cases apart.
_PROBE = """\
import locale
encoding = locale.getpreferredencoding(False)
try:
    bytes.fromhex("e69db1e4baac").decode(encoding)
    print("decodes", encoding)
except UnicodeDecodeError:
    print("raises", encoding)
"""

_CHILD = """\
import json, sys
from aelfrice.core_gate_background import label_batch
from aelfrice.store import MemoryStore
db, batch_id, cli = sys.argv[1:4]
store = MemoryStore(db)
try:
    record = label_batch(store, batch_id, claude=cli)
finally:
    store.close()
sys.stdout.write(json.dumps(record))
"""


@pytest.mark.timeout(120)  # #1307: two interpreter spawns, one of them nested
def test_the_reply_decodes_as_utf8_under_a_non_utf8_locale(tmp_path: Path) -> None:
    """Killed by: dropping `encoding="utf-8"` from the runner call. The stub's
    reply carries bytes the locale encoding can't decode. Without the pin,
    on POSIX the decode raises `UnicodeDecodeError`, a `ValueError` the run
    guard doesn't catch, and `label_batch` reports `store_error`; on Windows
    the decode error ends the reader thread, `stdout` is None, and the run is
    `run_failed`. Either way no label is cached."""
    env = {**os.environ, **_LOCALE_ENV}
    probe = subprocess.run(
        [sys.executable, "-c", _PROBE], env=env, capture_output=True,
        text=True, encoding="utf-8", check=False, timeout=60,
    )
    assert probe.stdout.split()[:1] == ["raises"], (
        "the child's locale encoding decodes the reply bytes, so this test "
        f"exercises nothing the fix affects: {probe.stdout!r} {probe.stderr!r}"
    )
    db = tmp_path / "memory.db"
    batch_id = _open_batch(db)
    cli = _write_stub(tmp_path)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(db), batch_id, cli],
        env=env, capture_output=True, text=True, encoding="utf-8", check=False, timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    record = json.loads(proc.stdout)
    assert record["outcome"] == bg.OUTCOME_ACCEPTED, record
    assert _labels(db) == {"hash-w1": "A"}


# --- the spawn flags ------------------------------------------------------------


class FakePopen:
    seen: dict[str, Any] = {}

    def __init__(self, argv: list[str], **kw: Any) -> None:
        FakePopen.seen = {"argv": argv, "kw": kw}


def test_spawn_starts_in_a_console_with_no_window_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: passing `start_new_session` on Windows, where Popen ignores
    it; dropping `CREATE_NO_WINDOW` or `CREATE_NEW_PROCESS_GROUP`, which
    would leave a console window or a shared process group; adding
    `DETACHED_PROCESS`, next to which Windows ignores `CREATE_NO_WINDOW`,
    so the python.exe a venv's launcher starts would open a visible
    console; or resolving the flags through a name `subprocess` has only on
    Windows without the fallback."""
    monkeypatch.setattr(bg, "_is_win32", lambda: True)
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    assert bg.spawn("widgetbatch", db_path="/x/memory.db", claude=CLAUDE) is True
    kw = FakePopen.seen["kw"]
    assert kw["creationflags"] == CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
    assert not kw["creationflags"] & DETACHED_PROCESS
    assert "start_new_session" not in kw
    for stream in ("stdin", "stdout", "stderr"):
        assert kw[stream] is subprocess.DEVNULL, stream


def test_spawn_starts_a_new_session_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed by: passing `creationflags` on POSIX, where a non-zero value
    makes Popen raise and the hook report a failed spawn."""
    monkeypatch.setattr(bg, "_is_win32", lambda: False)
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    assert bg.spawn("widgetbatch", db_path="/x/memory.db", claude=CLAUDE) is True
    kw = FakePopen.seen["kw"]
    assert kw["start_new_session"] is True
    assert "creationflags" not in kw


def test_the_flags_resolve_through_subprocess_when_it_has_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: reading only the module's own table and never `subprocess`,
    so a platform whose values differ would get the wrong bits."""
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x40, raising=False)
    assert bg._creationflags("CREATE_NO_WINDOW") == 0x40  # pyright: ignore[reportPrivateUsage]
    monkeypatch.delattr(subprocess, "CREATE_NO_WINDOW", raising=False)
    assert bg._creationflags("CREATE_NO_WINDOW") == CREATE_NO_WINDOW  # pyright: ignore[reportPrivateUsage]


def test_the_platform_probe_reads_sys_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Killed by: a probe that never reports Windows, which would turn the
    flags above off on the one host they are for. The tests above replace
    the probe, so only this one reaches it on a POSIX runner."""
    monkeypatch.setattr(bg.sys, "platform", "win32")
    assert bg._is_win32() is True  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(bg.sys, "platform", "linux")
    assert bg._is_win32() is False  # pyright: ignore[reportPrivateUsage]


class FakeRun:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, argv: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append({"argv": argv, **kw})
        reply = json.dumps({"type": "result", "result": json.dumps([{"index": 0, "label": "A"}])})
        return subprocess.CompletedProcess(argv, 0, reply, "")


@pytest.mark.parametrize("win32", [True, False])
def test_the_cli_run_hides_its_window_on_windows_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, win32: bool,
) -> None:
    """Killed by: running the CLI without `CREATE_NO_WINDOW` on Windows,
    which would leave the CLI's window to whatever console this process
    has; or passing `creationflags` on POSIX, where Popen rejects it."""
    monkeypatch.setattr(bg, "_is_win32", lambda: win32)
    db = tmp_path / "memory.db"
    batch_id = _open_batch(db)
    run = FakeRun()
    store = MemoryStore(str(db))
    try:
        record = bg.label_batch(store, batch_id, claude=CLAUDE, run=run, env={})
    finally:
        store.close()
    assert record["outcome"] == bg.OUTCOME_ACCEPTED
    (call,) = run.calls
    if win32:
        assert call["creationflags"] == CREATE_NO_WINDOW
    else:
        assert "creationflags" not in call


# --- the timeout through a claude.cmd shim ---------------------------------------


class HungShimPopen:
    """A `Popen` for a `claude.cmd` run that never finishes.

    It stands for cmd.exe, whose node.exe child holds the pipes: until the
    tree is killed, `communicate` with a timeout times out, and one with no
    timeout would block until node exits, so it raises instead. With
    `tree_dies`, a `taskkill /T` for its pid closes the pipes; without it,
    they stay open. A `taskkill.exe` instance returns at once.
    """

    events: list[tuple[str, Any]] = []
    tree_dies = True

    def __init__(self, argv: list[str], **kw: Any) -> None:
        self.argv = argv
        self.args = argv
        self.kw = kw
        self.stdin = io.StringIO()
        self.pid = 4242
        self.returncode: int | None = None
        self.tree_killed = False
        HungShimPopen.events.append(("popen", argv))

    def __enter__(self) -> HungShimPopen:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def communicate(
        self, input: str | None = None, timeout: float | None = None,
    ) -> tuple[str | None, str | None]:
        HungShimPopen.events.append(("communicate", timeout))
        if self.argv[0].lower().endswith("taskkill.exe"):
            self.returncode = 0
            if HungShimPopen.tree_dies:
                for name, arg in HungShimPopen.events:
                    if name == "kill_target":
                        arg.tree_killed = True
            return None, None
        HungShimPopen.events.append(("kill_target", self))
        if timeout is None:
            raise AssertionError("communicate() with no timeout waits for node.exe to exit")
        if self.tree_killed:
            return "", ""
        raise subprocess.TimeoutExpired(self.argv, timeout)

    def kill(self) -> None:
        HungShimPopen.events.append(("kill", self.argv[0]))
        self.returncode = 1

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


@pytest.mark.parametrize("tree_dies", [True, False])
def test_a_timeout_through_a_cmd_shim_ends_the_tree_and_returns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tree_dies: bool,
) -> None:
    """Killed by: running the CLI through `subprocess.run`, whose Windows
    timeout path kills cmd.exe only and then calls `communicate()` with no
    timeout, waiting on node.exe for as long as the hung run lasts; ending
    the run without `taskkill /F /T` on its pid; a bare `taskkill`, which
    `CreateProcess` would look for in the project first; or a final wait
    with no timeout, which `tree_dies=False` (a `taskkill` that missed the
    tree) would turn into the same hang. `subprocess._mswindows` is set so
    `subprocess.run` takes its Windows branch here."""
    db = tmp_path / "memory.db"
    batch_id = _open_batch(db)
    HungShimPopen.events = []
    HungShimPopen.tree_dies = tree_dies
    monkeypatch.setattr(bg, "_is_win32", lambda: True)
    monkeypatch.setenv("SystemRoot", "C:\\WidgetWin")
    monkeypatch.setattr(subprocess, "Popen", HungShimPopen)
    monkeypatch.setattr(subprocess, "_mswindows", True)
    store = MemoryStore(str(db))
    try:
        record = bg.label_batch(store, batch_id, claude="C:\\widget\\claude.cmd", env={})
    finally:
        store.close()
    assert record["outcome"] == bg.OUTCOME_RUN_FAILED, record
    assert record["error"] == "TimeoutExpired"
    timeouts = [arg for name, arg in HungShimPopen.events if name == "communicate"]
    assert None not in timeouts
    assert timeouts[0] == bg.RUNNER_TIMEOUT_SECONDS
    taskkill = [arg for name, arg in HungShimPopen.events
                if name == "popen" and arg[0].lower().endswith("taskkill.exe")]
    assert taskkill == [[
        ntpath.join("C:\\WidgetWin", "System32", "taskkill.exe"), "/F", "/T", "/PID", "4242",
    ]]
    order = [name if name != "popen" else arg[0] for name, arg in HungShimPopen.events
             if name in ("popen", "kill")]
    assert order.index(taskkill[0][0]) < order.index("kill")
    assert _labels(db) == {}


def test_the_settings_file_is_removed_after_a_timeout_through_a_cmd_shim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: a timeout path that never returns, which keeps the
    `finally` that removes the file from running."""
    db = tmp_path / "memory.db"
    batch_id = _open_batch(db)
    HungShimPopen.events = []
    HungShimPopen.tree_dies = False
    monkeypatch.setattr(bg, "_is_win32", lambda: True)
    monkeypatch.setattr(subprocess, "Popen", HungShimPopen)
    monkeypatch.setattr(subprocess, "_mswindows", True)
    store = MemoryStore(str(db))
    try:
        record = bg.label_batch(store, batch_id, claude="C:\\widget\\claude.cmd", env={})
    finally:
        store.close()
    assert record["outcome"] == bg.OUTCOME_RUN_FAILED, record
    (argv,) = [arg for name, arg in HungShimPopen.events
               if name == "popen" and not arg[0].lower().endswith("taskkill.exe")]
    assert not Path(argv[argv.index("--settings") + 1]).exists()


# The longest a `_FullPipe` write blocks before the test gives up on it.
_FULL_PIPE_BLOCK_SECONDS = 5.0


class _FullPipe:
    """A stdin pipe whose buffer is full: a write blocks until the reader dies."""

    def __init__(self) -> None:
        self.broken = threading.Event()
        self.closed = threading.Event()

    def write(self, data: str) -> int:
        if self.broken.wait(_FULL_PIPE_BLOCK_SECONDS):
            raise BrokenPipeError(32, "Broken pipe")
        raise AssertionError("the stdin write blocked, and no kill ended it")

    def close(self) -> None:
        self.closed.set()


class UnreadStdinPopen:
    """A `Popen` for a run that never reads stdin and never finishes.

    Its `communicate` writes `input` before it starts the timeout, as
    CPython 3.12's Windows `communicate` does, so a prompt larger than the
    pipe's buffer blocks there. `kill` breaks the pipe.
    """

    last: UnreadStdinPopen | None = None

    def __init__(self, argv: list[str], **kw: Any) -> None:
        self.argv = argv
        self.args = argv
        self.pid = 4343
        self.returncode: int | None = None
        self.pipe = _FullPipe()
        self.stdin: _FullPipe | None = self.pipe
        UnreadStdinPopen.last = self

    def communicate(
        self, input: str | None = None, timeout: float | None = None,
    ) -> tuple[str, str]:
        if self.stdin is not None:
            if input:
                self.stdin.write(input)
            self.stdin.close()
        if self.returncode is not None:
            return "", ""
        assert timeout is not None, "communicate() with no timeout waits for the run"
        self.pipe.broken.wait(timeout)
        raise subprocess.TimeoutExpired(self.argv, timeout)

    def kill(self) -> None:
        self.returncode = 1
        self.pipe.broken.set()


def test_a_run_that_never_reads_its_prompt_still_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Killed by: handing the prompt to `communicate`, which on Windows under
    CPython 3.12 writes it before the timeout starts, so a run that never
    reads blocks the call past `timeout` until the pipe breaks."""
    monkeypatch.setattr(bg, "_is_win32", lambda: False)
    monkeypatch.setattr(subprocess, "Popen", UnreadStdinPopen)
    # The fake's pid is made up, and a real group may hold that number.
    monkeypatch.setattr(bg.os, "killpg", lambda _pid, _sig: None, raising=False)
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        bg.run_bounded(["widget-cli"], input="w" * 65536, timeout=0.2)
    assert time.monotonic() - started < _FULL_PIPE_BLOCK_SECONDS
    proc = UnreadStdinPopen.last
    assert proc is not None and proc.stdin is None
    assert proc.pipe.closed.wait(_FULL_PIPE_BLOCK_SECONDS), "the stdin thread never closed the pipe"


# --- the stub CLI: argv, settings file, and the real spawn path ----------------


def _run_against_the_stub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve the stub through `find_claude`, run one batch, check what arrived."""
    _write_stub(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    if WIN32:
        monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT;.CMD")
    cli = _REAL_FIND_CLAUDE()
    assert cli is not None, "find_claude did not resolve the stub"
    assert Path(cli).resolve().parent == tmp_path.resolve()
    assert Path(cli).name.lower() == ("claude.cmd" if WIN32 else "claude")
    db = tmp_path / "memory.db"
    batch_id = _open_batch(db)
    store = MemoryStore(str(db))
    try:
        record = bg.label_batch(store, batch_id, claude=cli)
    finally:
        store.close()
    assert record["outcome"] == bg.OUTCOME_ACCEPTED, record
    assert _labels(db) == {"hash-w1": "A"}
    argv = json.loads((tmp_path / "argv.json").read_text(encoding="utf-8"))
    settings = argv[argv.index("--settings") + 1]
    assert argv == bg.runner_argv(cli, settings)[1:]
    assert (tmp_path / "settings-copy.json").read_text(encoding="utf-8") == bg.RUNNER_SETTINGS_JSON
    assert not Path(settings).exists()
    assert "widgetwin w1" in (tmp_path / "prompt.txt").read_text(encoding="utf-8")


@pytest.mark.skipif(not WIN32, reason="the cmd.exe shim path; the POSIX variant is below")
@pytest.mark.timeout(120)  # #1307: a real interpreter spawn through cmd.exe
def test_a_cmd_shim_receives_the_argv_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Windows leg. Killed by: an argument that cmd.exe reshapes on the
    way through `%*`, a `--settings` file the CLI can't read, or a labeler
    that can't run a `.cmd` at all. The `--system-prompt` value carries
    spaces and an apostrophe, and `--tools` is the empty string."""
    _run_against_the_stub(tmp_path, monkeypatch)


@pytest.mark.skipif(WIN32, reason="the POSIX shim path; the Windows variant is above")
@pytest.mark.timeout(120)  # #1307: a real interpreter spawn through sh
def test_a_posix_shim_receives_the_argv_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The required legs' arm of the test above: the same stub behind `sh`,
    so the argv and settings-file assertions run on every pull request."""
    _run_against_the_stub(tmp_path, monkeypatch)


@pytest.mark.skipif(not WIN32, reason="the real Windows spawn; POSIX is covered in #1741's tests")
@pytest.mark.timeout(120)  # #1307: a real detached child
def test_the_detached_child_runs_the_stub_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Windows leg's spawn path with no fake: Popen accepts the flag set,
    the detached child finds the store, runs the `.cmd`, and logs `accepted`.
    Killed by: a flag combination `CreateProcess` rejects, or a child that
    can't reach the store or the shim from a detached start."""
    cli = _write_stub(tmp_path)
    store = tmp_path / "store" / "memory.db"
    store.parent.mkdir()
    batch_id = _open_batch(store)
    monkeypatch.setenv("AELFRICE_DB", str(store))
    assert bg.spawn(batch_id, db_path=str(store), claude=cli) is True
    log = store.parent / bg.LOG_FILENAME
    text = ""
    deadline = time.monotonic() + 90
    # The child opens the log before it writes the line, so wait for the
    # newline that ends the record rather than for the file.
    while not text.endswith("\n") and time.monotonic() < deadline:
        time.sleep(0.2)
        text = log.read_text(encoding="utf-8") if log.exists() else ""
    (rec,) = [json.loads(x) for x in text.splitlines()]
    assert rec["outcome"] == bg.OUTCOME_ACCEPTED, rec
    assert _labels(store) == {"hash-w1": "A"}
