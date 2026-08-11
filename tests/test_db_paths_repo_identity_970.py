"""Repo-identity derivation + _open_store injection for #970.

`project_context` is stamped with a repo identity derived from the
store's on-disk location (no extra git fork). These tests cover the
derivation helpers and that `_open_store()` threads the identity into
the store so writes get stamped.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice import db_paths
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief


def _b(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"{bid}-hash",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-06-18T00:00:00Z",
        last_retrieved_at=None,
    )


def test_identity_is_deterministic_and_formatted() -> None:
    git_dir = Path("/home/u/projects/myrepo/.git")
    ident = db_paths._identity_from_git_common_dir(git_dir)
    assert ident == db_paths._identity_from_git_common_dir(git_dir)
    assert ident.startswith("myrepo-")
    assert len(ident.split("-")[-1]) == 8


def test_identity_distinguishes_same_named_repos() -> None:
    a = db_paths._identity_from_git_common_dir(Path("/a/myrepo/.git"))
    b = db_paths._identity_from_git_common_dir(Path("/b/myrepo/.git"))
    assert a != b
    assert a.startswith("myrepo-") and b.startswith("myrepo-")


def test_repo_identity_from_db_path_repo_layout() -> None:
    p = Path("/home/u/proj/.git/aelfrice/memory.db")
    expected = db_paths._identity_from_git_common_dir(Path("/home/u/proj/.git"))
    assert db_paths.repo_identity_from_db_path(p) == expected


def test_repo_identity_from_db_path_home_fallback_is_empty() -> None:
    assert db_paths.repo_identity_from_db_path(Path.home() / ".aelfrice" / "memory.db") == ""


def test_repo_identity_from_db_path_memory_is_empty() -> None:
    assert db_paths.repo_identity_from_db_path(Path(":memory:")) == ""


def test_repo_identity_none_outside_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db_paths, "_git_common_dir", lambda: None)
    assert db_paths.repo_identity() == ""


def test_repo_identity_uses_git_common_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        db_paths, "_git_common_dir", lambda: Path("/x/somerepo/.git"),
    )
    assert db_paths.repo_identity() == db_paths._identity_from_git_common_dir(
        Path("/x/somerepo/.git"),
    )


def test_git_common_dir_none_on_undecodable_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Undecodable git output falls back; it does not become a directory.

    `_git_common_dir` decodes strict (#1441), so `subprocess.run` raises
    `UnicodeDecodeError` — a `ValueError`, caught by neither `OSError`
    nor `TimeoutExpired`. Escaping here is not cosmetic: `db_path()`
    appends to whatever this returns and `_open_store()` mkdirs it, so
    anything other than None would create a second store on disk.
    """
    def _boom(*_args: object, **_kwargs: object) -> object:
        raise UnicodeDecodeError("utf-8", b"\xe6", 0, 1, "bad")

    monkeypatch.setattr(db_paths.subprocess, "run", _boom)
    assert db_paths._git_common_dir() is None


def test_git_common_dir_decodes_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    """The kwargs say strict, not replace — the arm above stays reachable.

    Pinned separately because `errors="replace"` never raises: the test
    above would then pass on a mock that no production call can produce.
    """
    seen: dict[str, object] = {}

    def _capture(*_args: object, **kwargs: object) -> object:
        seen.update(kwargs)
        raise FileNotFoundError("git")

    monkeypatch.setattr(db_paths.subprocess, "run", _capture)
    assert db_paths._git_common_dir() is None
    assert seen.get("encoding") == "utf-8"
    assert seen.get("errors") == "strict"


def test_open_store_stamps_writes_in_repo_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "aelfrice" / "memory.db"  # repo-store layout
    monkeypatch.setenv("AELFRICE_DB", str(db))
    expected = db_paths.repo_identity_from_db_path(db)
    assert expected != ""
    store = db_paths._open_store()
    store.insert_belief(_b("b1"))
    got = store.get_belief("b1")
    assert got is not None
    assert got.project_context == expected


def test_open_store_no_identity_for_non_layout_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "memory.db"  # not under an 'aelfrice' dir
    monkeypatch.setenv("AELFRICE_DB", str(db))
    store = db_paths._open_store()
    store.insert_belief(_b("b1"))
    got = store.get_belief("b1")
    assert got is not None
    assert got.project_context == ""
