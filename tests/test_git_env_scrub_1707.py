"""#1707: the suite clears git's location variables before any test runs.

A git hook exports `GIT_DIR`, so a suite run from inside one used to point
every tmp-dir `git init`, and every git-dir store lookup, at the outer
repository: roughly 130 tests failed or errored, and an `aelfrice/` store
appeared in the outer `.git`.

Checking `os.environ` in this process proves nothing: CI never exports
these variables, so such a check passes with or without the scrub. Instead
the guard runs a second pytest with every one of them exported, against
`test_the_suite_sees_no_git_location_variable` below, which then has to
pass inside that run.

The list is written out here rather than imported from conftest: checking
conftest's list against itself would pass after a name was dropped from it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

#: Every git variable that relocates the repository or its parts.
EXPECTED_CLEARED = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
)

_HERE = Path(__file__).resolve()


def test_the_suite_sees_no_git_location_variable() -> None:
    """Holds in every run; the guard below makes it hold under export."""
    present = [v for v in EXPECTED_CLEARED if v in os.environ]
    assert present == [], f"git location variables leaked into the suite: {present}"


@pytest.mark.timeout(120)
def test_every_exported_location_variable_is_cleared(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    outer.mkdir()
    clean = {k: v for k, v in os.environ.items() if k not in EXPECTED_CLEARED}
    subprocess.run(["git", "init", "-q", str(outer)], check=True, timeout=30, env=clean)
    git_dir = outer / ".git"
    exported = {
        "GIT_DIR": str(git_dir),
        "GIT_WORK_TREE": str(outer),
        "GIT_INDEX_FILE": str(git_dir / "index"),
        "GIT_COMMON_DIR": str(git_dir),
        "GIT_OBJECT_DIRECTORY": str(git_dir / "objects"),
        "GIT_CEILING_DIRECTORIES": str(tmp_path),
        "GIT_DISCOVERY_ACROSS_FILESYSTEM": "1",
    }
    assert set(exported) == set(EXPECTED_CLEARED)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"{_HERE}::test_the_suite_sees_no_git_location_variable"],
        cwd=_HERE.parents[1], env={**clean, **exported}, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=90,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
