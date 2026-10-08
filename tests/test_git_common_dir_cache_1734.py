"""`_git_common_dir` asks git once per cwd and discovery environment (#1734)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from aelfrice import db_paths


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True, timeout=30)


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


@pytest.mark.timeout(60)
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


@pytest.mark.timeout(60)
@pytest.mark.usefixtures("no_git_env")
def test_a_none_result_is_cached_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lookups: list[str],
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    assert db_paths._git_common_dir() is None
    assert db_paths._git_common_dir() is None
    assert len(lookups) == 1


@pytest.mark.timeout(60)
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


@pytest.mark.timeout(60)
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


@pytest.mark.parametrize("name", [
    "GIT_DIR",
    "GIT_COMMON_DIR",
    "GIT_WORK_TREE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
])
@pytest.mark.usefixtures("no_git_env")
def test_each_discovery_variable_is_part_of_the_key(
    name: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str | None] = []

    def recording() -> Path | None:
        seen.append(db_paths.os.environ.get(name))
        return None

    monkeypatch.setattr(db_paths, "_lookup_git_common_dir", recording)
    # Each value is looked up twice; only the first of each reaches git.
    db_paths._git_common_dir()
    db_paths._git_common_dir()
    monkeypatch.setenv(name, "")
    db_paths._git_common_dir()
    db_paths._git_common_dir()
    monkeypatch.setenv(name, "x")
    db_paths._git_common_dir()
    db_paths._git_common_dir()
    assert seen == [None, "", "x"]


@pytest.mark.timeout(60)
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

    calls: list[bool] = []

    def lookup() -> Path | None:
        calls.append(True)
        return None

    monkeypatch.setattr(db_paths.os, "getcwd", gone)
    monkeypatch.setattr(db_paths, "_lookup_git_common_dir", lookup)
    assert db_paths._git_common_dir() is None
    assert db_paths._git_common_dir() is None
    assert calls == [True, True]
    assert db_paths._git_common_dir_cache == {}


@pytest.mark.usefixtures("no_git_env")
def test_a_timed_out_lookup_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def lookup() -> Path | None:
        calls.append(True)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("git", 5)
        return tmp_path / ".git"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(db_paths, "_lookup_git_common_dir", lookup)
    assert db_paths._git_common_dir() is None
    assert db_paths._git_common_dir() == tmp_path / ".git"
    assert db_paths._git_common_dir() == tmp_path / ".git"
    assert calls == [True, True]


def test_the_lookup_raises_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(*_args: object, **_kwargs: object) -> object:
        raise subprocess.TimeoutExpired("git", 5)

    monkeypatch.setattr(db_paths.subprocess, "run", slow)
    with pytest.raises(subprocess.TimeoutExpired):
        db_paths._lookup_git_common_dir()
