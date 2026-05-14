"""Two-worktree concurrent-access tests for `.git/aelfrice/memory.db`.

After v1.1.0 phase 1 (#88), two worktrees of the same repo resolve to
the same DB via `git rev-parse --git-common-dir`. SQLite WAL + a
`busy_timeout` are the documented mechanism for safe concurrent access;
this module verifies both are on AND that concurrent writes from
separate processes succeed without `database is locked` errors.

Multiprocessing (not threading) is the right substrate: each worktree
in production runs `aelf` in its own process with its own sqlite
connection. Threads sharing one process would test a different
scenario (intra-process connection sharing) that's not how aelfrice
is actually used.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
from pathlib import Path

import pytest

from aelfrice.cli import db_path
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    Belief,
)
from aelfrice.store import MemoryStore


# Number of beliefs each worktree writes. Picked to:
# - exceed any plausible WAL checkpoint without forcing one
# - run in well under the 5s pytest timeout per test
# - exercise enough sequential commits to surface contention if it exists
_WRITES_PER_WORKER: int = 20


def _init_repo(repo: Path) -> None:
    """Create a fresh git repo with one commit, ready for `git worktree add`."""
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "README").write_text("seed", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git",
         "-c", "user.email=test@example.invalid",
         "-c", "user.name=test",
         "-c", "commit.gpgsign=false",
         "commit", "-q", "-m", "seed"],
        cwd=repo, check=True,
    )


def _add_worktree(repo: Path, wt: Path, branch: str) -> None:
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", branch, str(wt)],
        cwd=repo, check=True,
    )


def _belief(worker_id: str, n: int) -> Belief:
    bid = f"{worker_id}-{n:04d}"
    content = (
        f"belief {n} written from worker {worker_id} during the "
        f"worktree concurrency test fixture"
    )
    now = "2026-04-27T00:00:00Z"
    return Belief(
        id=bid,
        content=content,
        content_hash=f"hash-{bid}",
        alpha=1.0,
        beta=0.5,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at=now,
        last_retrieved_at=None,
    )


def _writer_worker(cwd: str, db: str, worker_id: str, n: int) -> int:
    """Run inside a forked process. Open the DB, write n beliefs, close."""
    os.chdir(cwd)
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(db)
    try:
        for i in range(n):
            store.insert_belief(_belief(worker_id, i))
    finally:
        store.close()
    return n


@pytest.fixture
def _two_worktrees(tmp_path: Path):
    """Create a repo + a second worktree on `feature` branch."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    wt = tmp_path / "worktree-feature"
    _add_worktree(repo, wt, "feature")
    return repo, wt


# --- Pragmas / configuration ----------------------------------------


def test_wal_mode_is_on(tmp_path: Path) -> None:
    """A fresh on-disk store has journal_mode=WAL."""
    db = tmp_path / "wal.db"
    store = MemoryStore(str(db))
    try:
        cur = store._conn.execute("PRAGMA journal_mode")  # type: ignore[attr-defined]
        mode = cur.fetchone()[0]
    finally:
        store.close()
    assert mode == "wal"


def test_busy_timeout_is_set(tmp_path: Path) -> None:
    """busy_timeout is non-zero (must wait for a write lock, not fail)."""
    db = tmp_path / "bt.db"
    store = MemoryStore(str(db))
    try:
        cur = store._conn.execute("PRAGMA busy_timeout")  # type: ignore[attr-defined]
        timeout = cur.fetchone()[0]
    finally:
        store.close()
    assert timeout >= 1000, f"busy_timeout too small: {timeout}ms"


# --- Identity: worktrees share one DB --------------------------------


def test_two_worktrees_resolve_to_same_db_path(_two_worktrees,
                                                monkeypatch) -> None:
    repo, wt = _two_worktrees
    monkeypatch.delenv("AELFRICE_DB", raising=False)

    monkeypatch.chdir(repo)
    main_db = db_path()
    monkeypatch.chdir(wt)
    wt_db = db_path()
    assert main_db == wt_db


# --- Concurrent write correctness ------------------------------------


@pytest.mark.timeout(30)
def test_two_worktrees_concurrent_writes_succeed(_two_worktrees) -> None:
    """Two processes writing to the shared DB simultaneously: no
    `database is locked` errors, all rows land. WAL + busy_timeout
    is the contract under test."""
    repo, wt = _two_worktrees
    db = str(repo / ".git" / "aelfrice" / "memory.db")

    ctx = mp.get_context("spawn")
    p1 = ctx.Process(
        target=_writer_worker,
        args=(str(repo), db, "A", _WRITES_PER_WORKER),
    )
    p2 = ctx.Process(
        target=_writer_worker,
        args=(str(wt), db, "B", _WRITES_PER_WORKER),
    )

    p1.start()
    p2.start()
    p1.join(timeout=20)
    p2.join(timeout=20)

    assert p1.exitcode == 0, f"writer A exit code: {p1.exitcode}"
    assert p2.exitcode == 0, f"writer B exit code: {p2.exitcode}"

    # Final count: every row from both workers landed.
    store = MemoryStore(db)
    try:
        n = store.count_beliefs()
    finally:
        store.close()
    assert n == 2 * _WRITES_PER_WORKER


@pytest.mark.timeout(30)
def test_concurrent_writes_under_repeated_runs_do_not_corrupt(
    _two_worktrees,
) -> None:
    """Run the concurrent writers three times back to back. Detects
    flakes that wouldn't surface in one trial — busy_timeout exhaustion,
    WAL checkpoint races, ungraceful connection close. Final count
    must be exact."""
    repo, wt = _two_worktrees
    db = str(repo / ".git" / "aelfrice" / "memory.db")
    ctx = mp.get_context("spawn")

    rounds = 3
    expected = 0
    for r in range(rounds):
        p1 = ctx.Process(
            target=_writer_worker,
            args=(str(repo), db, f"A{r}", _WRITES_PER_WORKER),
        )
        p2 = ctx.Process(
            target=_writer_worker,
            args=(str(wt), db, f"B{r}", _WRITES_PER_WORKER),
        )
        p1.start()
        p2.start()
        p1.join(timeout=20)
        p2.join(timeout=20)
        assert p1.exitcode == 0 and p2.exitcode == 0
        expected += 2 * _WRITES_PER_WORKER

    store = MemoryStore(db)
    try:
        n = store.count_beliefs()
    finally:
        store.close()
    assert n == expected
