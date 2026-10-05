"""#1707: the suite clears git's location variables before any test runs.

A git hook exports `GIT_DIR`, so a suite run from inside one used to point
every tmp-dir `git init`, and every git-dir store lookup, at the outer
repository: 129 tests failed and an `aelfrice/` store appeared in the
outer `.git`.

Checking `os.environ` in this process proves nothing: CI never exports
`GIT_DIR`, so such a check passes with or without the scrub. Instead the
guard runs a second pytest with the variables exported, against
`test_the_suite_sees_no_git_location_variable` below, which then has to
pass inside that run.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import GIT_LOCATION_VARS

_HERE = Path(__file__).resolve()


def test_the_suite_sees_no_git_location_variable() -> None:
    """Holds in every run; the guard below makes it hold under export."""
    present = [v for v in GIT_LOCATION_VARS if v in os.environ]
    assert present == [], f"git location variables leaked into the suite: {present}"


@pytest.mark.timeout(120)
def test_an_exported_git_dir_does_not_reach_a_test(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    outer.mkdir()
    subprocess.run(["git", "init", "-q", str(outer)], check=True, timeout=30)
    env = {**os.environ, "GIT_DIR": str(outer / ".git"), "GIT_WORK_TREE": str(outer)}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"{_HERE}::test_the_suite_sees_no_git_location_variable"],
        cwd=_HERE.parents[1], env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=90,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert not (outer / ".git" / "aelfrice").exists()
