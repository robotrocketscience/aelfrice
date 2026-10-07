"""`_git_common_dir` asks git once per cwd and discovery environment (#1734)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aelfrice import db_paths


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)


@pytest.fixture
def lookups(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the cwd of every real git-common-dir lookup."""
    seen: list[str] = []
    real = db_paths._lookup_git_common_dir

    def counting() -> Path | None:
        seen.append(str(Path.cwd()))
        return real()

    monkeypatch.setattr(db_paths, "_lookup_git_common_dir", counting)
    return seen


@pytest.fixture
def no_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in db_paths._GIT_DISCOVERY_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("no_git_env")
def test_repeated_calls_in_one_cwd_run_git_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lookups: list[str],
) -> None:
    _git_init(tmp_path)
    monkeypatch.chdir(tmp_path)
    first = db_paths._git_common_dir()
    for _ in range(5):
        assert db_paths._git_common_dir() == first
    assert first == (tmp_path / ".git").resolve()
    assert len(lookups) == 1


@pytest.mark.usefixtures("no_git_env")
def test_a_none_result_is_cached_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lookups: list[str],
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    assert db_paths._git_common_dir() is None
    assert db_paths._git_common_dir() is None
    assert len(lookups) == 1


@pytest.mark.usefixtures("no_git_env")
def test_two_cwds_do_not_share_a_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lookups: list[str],
) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _git_init(a)
    _git_init(b)
    monkeypatch.chdir(a)
    from_a = db_paths._git_common_dir()
    monkeypatch.chdir(b)
    from_b = db_paths._git_common_dir()
    assert from_a == (a / ".git").resolve()
    assert from_b == (b / ".git").resolve()
    assert len(lookups) == 2


@pytest.mark.usefixtures("no_git_env")
def test_a_git_dir_variable_is_part_of_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lookups: list[str],
) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _git_init(a)
    _git_init(b)
    monkeypatch.chdir(a)
    assert db_paths._git_common_dir() == (a / ".git").resolve()
    monkeypatch.setenv("GIT_DIR", str(b / ".git"))
    assert db_paths._git_common_dir() == (b / ".git").resolve()
    assert len(lookups) == 2


@pytest.mark.usefixtures("no_git_env")
def test_clearing_the_cache_sees_a_new_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lookups: list[str],
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    assert db_paths._git_common_dir() is None
    _git_init(tmp_path)
    db_paths.clear_git_common_dir_cache()
    assert db_paths._git_common_dir() == (tmp_path / ".git").resolve()
    assert len(lookups) == 2


def test_an_unreadable_cwd_is_looked_up_and_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def gone() -> str:
        raise FileNotFoundError("cwd removed")

    monkeypatch.setattr(db_paths.os, "getcwd", gone)
    monkeypatch.setattr(db_paths, "_lookup_git_common_dir", lambda: None)
    assert db_paths._git_common_dir() is None
    assert db_paths._git_common_dir_cache == {}
