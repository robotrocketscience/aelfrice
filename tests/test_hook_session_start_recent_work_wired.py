"""Integration: <recent-work> wired into _build_session_start_subblock (#887)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from aelfrice.hook import (
    SESSION_START_SUBBLOCK_CLOSE,
    SESSION_START_SUBBLOCK_OPEN,
    _build_session_start_subblock,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_USER, Belief
from aelfrice.store import MemoryStore


def _run(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True)


def _init_repo(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _run(path, "init", "-q", "-b", "main")
    _run(path, "config", "user.email", "t@t")
    _run(path, "config", "user.name", "t")
    _run(path, "config", "commit.gpgsign", "false")


def _commit(repo: Path, name: str, subject: str) -> None:
    (repo / name).write_text(name)
    _run(repo, "add", name)
    _run(repo, "commit", "-q", "-m", subject)


def _seed_lock(db: Path, content: str) -> None:
    store = MemoryStore(str(db))
    try:
        store.insert_belief(Belief(
            id="L1",
            content=content,
            content_hash="h_L1",
            alpha=1.0,
            beta=1.0,
            type=BELIEF_FACTUAL,
            lock_level=LOCK_USER,
            locked_at="2026-01-01T00:00:00Z",
            created_at="2026-01-01T00:00:00Z",
            last_retrieved_at=None,
        ))
    finally:
        store.close()


def test_recent_work_appears_with_lock_pool(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit(repo, "base", "feat: base")
    _run(repo, "checkout", "-q", "-b", "feat/issue-887-recent-work")
    _commit(repo, "a", "feat: A (#887)")
    db = tmp_path / "memory.db"
    _seed_lock(db, "atomic commits beat batched")

    store = MemoryStore(str(db))
    try:
        block = _build_session_start_subblock(store, cwd=repo)
    finally:
        store.close()

    assert block.startswith(SESSION_START_SUBBLOCK_OPEN)
    assert block.endswith(SESSION_START_SUBBLOCK_CLOSE)
    assert "<locked>" in block
    assert "atomic commits beat batched" in block
    assert "<recent-work>" in block
    assert "<branch>feat/issue-887-recent-work</branch>" in block
    assert "feat: A (#887)" in block
    assert "<linked-issues>#887</linked-issues>" in block

    # Section ordering: locked, core, recent-work, close
    locked_idx = block.index("<locked>")
    core_idx = block.index("<core>")
    recent_idx = block.index("<recent-work>")
    close_idx = block.index(SESSION_START_SUBBLOCK_CLOSE)
    assert locked_idx < core_idx < recent_idx < close_idx


def test_recent_work_alone_emits_subblock(tmp_path: Path) -> None:
    """Empty store but live git repo: <recent-work> alone is enough to emit."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    _commit(repo, "x", "feat: x")
    db = tmp_path / "memory.db"

    store = MemoryStore(str(db))
    try:
        block = _build_session_start_subblock(store, cwd=repo)
    finally:
        store.close()

    assert SESSION_START_SUBBLOCK_OPEN in block
    assert "<recent-work>" in block
    # locked + core sections present but empty
    assert "<locked>\n</locked>" in block
    assert "<core>\n</core>" in block


def test_non_git_cwd_no_recent_work(tmp_path: Path) -> None:
    """Non-git cwd: <recent-work> omitted; if no locks/core, return ""."""
    db = tmp_path / "memory.db"
    store = MemoryStore(str(db))
    try:
        block = _build_session_start_subblock(store, cwd=tmp_path)
    finally:
        store.close()
    assert block == ""
