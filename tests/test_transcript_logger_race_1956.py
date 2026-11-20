"""#1956: two logger registrations started at once write one user row.

The host runs matching hooks in parallel, so two logger entries fire
together. The #968 duplicate guard reads the last line and then appends.
Without a lock around both steps, each process reads the tail before the
other appends, and both lines land.

The concurrent test widens that window on purpose so its result does not
depend on scheduler timing. Each child wraps `_read_last_turn_line`: after
its read it waits, up to `_READ_HOLD_SECONDS`, for the other child to
finish a read too. On the unlocked guard both reads happen, both see an
empty file, and both append. With the lock, the second child cannot read
until the first appends, so the first child's wait runs out, it appends,
and the second child reads that row and skips its own.

Both children stamp their row with one fixed timestamp, so the guard's
2-second duplicate window can't split them however late the second child
starts on a loaded runner.
"""
from __future__ import annotations

import concurrent.futures
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aelfrice import file_lock
from aelfrice import transcript_logger as tl
from aelfrice.print_mode import ENV_HOST_ENTRYPOINT

_READ_HOLD_SECONDS = 2.0
# Longer than the hold, so the waiting child gets the lock instead of
# timing out into the unlocked fallback.
_CHILD_LOCK_TIMEOUT_SECONDS = 15.0
_FIXED_TS = "2026-01-01T00:00:00+00:00"

_CHILD = r"""
import sys
import time
from pathlib import Path

from aelfrice import transcript_logger as tl

reads_dir = Path(sys.argv[1])
name = sys.argv[2]
hold = float(sys.argv[3])
tl._TURN_LOCK_TIMEOUT_SECONDS = float(sys.argv[4])
fixed_ts = sys.argv[5]
tl._now_iso = lambda: fixed_ts
original = tl._read_last_turn_line


def held_read(path):
    last = original(path)
    (reads_dir / name).touch()
    deadline = time.monotonic() + hold
    while len(list(reads_dir.iterdir())) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    return last


tl._read_last_turn_line = held_read
sys.exit(tl.main())
"""

_PAYLOAD = {
    "hook_event_name": "UserPromptSubmit",
    "session_id": "sess-race-1956",
    "prompt": "Synthetic prompt for the duplicate logging race.",
}


def _user_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [r for r in rows if r.get("role") == "user"]


def _child_env(root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop(ENV_HOST_ENTRYPOINT, None)
    env["HOME"] = str(root / "home")
    env["AELFRICE_DB"] = str(root / "memory.db")
    env["AELFRICE_USER_DB"] = str(root / "user.db")
    env["AELFRICE_TRANSCRIPTS_DIR"] = str(root / "transcripts")
    return env


def _run_child(
    name: str, reads_dir: Path, env: dict[str, str], cwd: Path,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable, "-c", _CHILD, str(reads_dir), name,
            str(_READ_HOLD_SECONDS), str(_CHILD_LOCK_TIMEOUT_SECONDS),
            _FIXED_TS,
        ],
        input=json.dumps(_PAYLOAD).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=cwd,
        check=False,
        timeout=60,
    )


@pytest.mark.timeout(120)
def test_two_concurrent_loggers_write_one_user_row(tmp_path: Path) -> None:
    (tmp_path / "home").mkdir()
    reads_dir = tmp_path / "reads"
    reads_dir.mkdir()
    env = _child_env(tmp_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_run_child, name, reads_dir, env, tmp_path)
            for name in ("a", "b")
        ]
        results = [f.result(timeout=90) for f in futures]
    for proc in results:
        assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    rows = _user_rows(tmp_path / "transcripts" / tl.TURNS_FILENAME)
    assert len(rows) == 1, f"expected 1 user row, got {len(rows)}"
    assert rows[0]["text"] == _PAYLOAD["prompt"]


@pytest.fixture
def tdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AELFRICE_TRANSCRIPTS_DIR", str(tmp_path))
    monkeypatch.setenv("AELFRICE_DB", str(tmp_path / "memory.db"))
    return tmp_path


@pytest.fixture
def held_lock(tdir: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Hold `turns.jsonl.lock` on a separate descriptor for the test.

    An advisory lock belongs to the open file, so the logger's own
    descriptor sees contention even inside this process.
    """
    if not file_lock.HAVE_ADVISORY_LOCKS:
        pytest.skip("no advisory lock backend on this host")
    monkeypatch.setattr(tl, "_TURN_LOCK_TIMEOUT_SECONDS", 0.05)
    fd = os.open(
        str(tdir / (tl.TURNS_FILENAME + ".lock")), os.O_CREAT | os.O_RDWR, 0o600,
    )
    file_lock.lock_exclusive(fd)
    try:
        yield fd
    finally:
        file_lock.unlock(fd)
        os.close(fd)


def _run_main() -> int:
    return tl.main(stdin=io.StringIO(json.dumps(_PAYLOAD)), stderr=io.StringIO())


def test_lock_timeout_still_appends_the_turn(tdir: Path, held_lock: int) -> None:
    """A lock held past the timeout must not cost the hook its turn."""
    assert _run_main() == 0
    assert len(_user_rows(tdir / tl.TURNS_FILENAME)) == 1


def test_lock_timeout_fallback_keeps_the_sequential_guard(
    tdir: Path, held_lock: int,
) -> None:
    """The unlocked fallback still drops a sequential re-fire (#968)."""
    assert _run_main() == 0
    assert _run_main() == 0
    assert len(_user_rows(tdir / tl.TURNS_FILENAME)) == 1


def test_lock_is_released_after_the_append(tdir: Path) -> None:
    """A second process must be able to take the lock once a turn lands."""
    assert _run_main() == 0
    fd = os.open(str(tdir / (tl.TURNS_FILENAME + ".lock")), os.O_RDWR)
    try:
        file_lock.try_lock_exclusive(fd)
        file_lock.unlock(fd)
    finally:
        os.close(fd)


def test_lock_wait_is_short() -> None:
    """The hook must not stall behind another process's lock."""
    assert 0 < tl._TURN_LOCK_TIMEOUT_SECONDS <= 0.5


@pytest.mark.timeout(60)
def test_locked_append_does_not_import_the_store_resolver(tmp_path: Path) -> None:
    """With the transcripts folder set, a logged turn stays off `db_paths`.

    The resolver drags in the store modules; importing it on the hot path
    cost about 9 ms per turn (#1956 review).
    """
    (tmp_path / "home").mkdir()
    code = (
        "import io, json, sys\n"
        "from aelfrice import transcript_logger as tl\n"
        "tl.main(stdin=io.StringIO(sys.argv[1]), stderr=io.StringIO())\n"
        "print('aelfrice.db_paths' in sys.modules)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, json.dumps(_PAYLOAD)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_child_env(tmp_path),
        cwd=tmp_path,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    assert _user_rows(tmp_path / "transcripts" / tl.TURNS_FILENAME)
    assert proc.stdout.decode().strip().splitlines()[-1] == "False"
