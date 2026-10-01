"""#1678 — a detached child spawned under the suite must not reach a real store.

`transcript_logger._spawn_background_ingest` runs `aelf ingest-transcript`
as a detached process. The child re-imports aelfrice and calls `db_path()`
for itself, so no `setattr` in the test process can steer it. Without a
pin, `db_path()` resolves the git common dir of the inherited cwd, which
under pytest is the contributor's checkout, and the child writes fixture
beliefs into `<repo>/.git/aelfrice/memory.db`. `test_transcript_round_trip`
did exactly that.

The fix is the `AELFRICE_DB` pin in conftest. `_restore_sandbox_store_pin`
sets it before and after every test, and `_sandbox_real_home` sets it once
per session for session-scoped fixtures. These guards cover the per-test
pin: removing it fails them. No session-scoped fixture spawns a child
today, so nothing here fails if only the session line goes. This guard
reproduces the child's view: a fresh interpreter that inherits the test
environment and runs from the repo root. It resolves the store path and
opens nothing, so it never writes.

Nothing here writes outside the sandbox. The real git common dir is only
read through `git rev-parse`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.timeout(60)

_REPO_ROOT = Path(__file__).resolve().parents[1]

_CHILD_SOURCE = "from aelfrice.db_paths import db_path; print(db_path())"


def _real_git_common_dir() -> Path | None:
    """The git common dir a cwd-resolved store would live under, or None."""
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return Path(result.stdout.strip()).resolve()


def test_detached_child_resolves_the_sandbox_store(
    _sandbox_real_home: Path,
) -> None:
    """A child inheriting the suite's environment resolves a sandbox store.

    The child is started the way `_spawn_background_ingest` starts its
    child: no `env=` argument, so it inherits `os.environ`, and a cwd
    inside the git work tree, so the git-dir branch of `db_path()` is live.
    """
    git_dir = _real_git_common_dir()
    if git_dir is None:
        pytest.skip(
            "the checkout is not a git work tree, so the git-dir branch "
            "of db_path() that leaked in #1678 is unreachable here"
        )

    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SOURCE],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    resolved = Path(result.stdout.strip()).resolve()

    assert not resolved.is_relative_to(git_dir), (
        f"a detached child resolves the real repo store {resolved}, under "
        f"{git_dir}. Pin AELFRICE_DB in conftest's _sandbox_real_home so "
        "every spawned `aelf` inherits a sandbox store (#1678)."
    )
    assert resolved.is_relative_to(_sandbox_real_home.resolve()), (
        f"a detached child resolves {resolved}, outside the session "
        f"sandbox {_sandbox_real_home}"
    )


# The next two tests run in file order, which pytest keeps. The first does
# what `test_wonder_skill_integration_e2e` does in its `finally`: it pops
# the variable with a bare `os.environ.pop`, which no `monkeypatch` undoes.
# The second checks that the pin is back before it runs. Without conftest's
# `_restore_sandbox_store_pin`, the pop deletes the session pin for the
# rest of the run, and every later spawn resolves the git-dir store.


def test_a_bare_pop_of_the_pin_simulates_a_leaky_test() -> None:
    """Delete the pin the way a leaky test does, without restoring it."""
    os.environ.pop("AELFRICE_DB", None)
    assert "AELFRICE_DB" not in os.environ


def test_the_pin_is_back_for_the_next_test(_sandbox_real_home: Path) -> None:
    """The test after a leaky one still sees the session pin."""
    assert os.environ.get("AELFRICE_DB") == str(
        _sandbox_real_home / "memory.db"
    ), (
        "an earlier test's bare os.environ change outlived it, so later "
        "spawns no longer inherit the sandbox store (#1678)"
    )


# The same pair for a bare assignment, which `tests/test_meta_beliefs.py`
# does: without the restore, every later test resolves this test's file.


def test_a_bare_assignment_of_the_pin_simulates_a_leaky_test(
    tmp_path: Path,
) -> None:
    """Repoint the pin the way a leaky test does, without restoring it."""
    os.environ["AELFRICE_DB"] = str(tmp_path / "leaked.db")


def test_the_pin_is_restored_after_an_assignment(
    _sandbox_real_home: Path,
) -> None:
    """The test after a leaky assignment still sees the session pin."""
    assert os.environ.get("AELFRICE_DB") == str(
        _sandbox_real_home / "memory.db"
    ), (
        "an earlier test's bare os.environ assignment outlived it, so later "
        "spawns resolve that test's store (#1678)"
    )
