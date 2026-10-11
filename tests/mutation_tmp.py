"""#1746: under mutmut, each pytest session removes its own temporary root.

pytest gives each session a numbered root, `pytest-of-<user>/pytest-N`, and
keeps only the last few, but it treats a root whose `.lock` file is still
there as live for three days. It removes that lock from an `atexit` handler.
mutmut runs a pytest session per mutant in a forked worker that never runs
`atexit` handlers, so every root survives with its lock: a local run of
about 3,800 mutants left about 3,850 of them, 20 GB, and filled the disk.

pytest's own `tmp_path_retention_policy = "none"` is not a fix: it sets the
keep count to zero, so every session, including one a test starts as a
subprocess, deletes every other session's root, the live ones included.

So a session removes its own root when it ends, and only under mutmut,
which sets `MUTANT_UNDER_TEST` in every process it runs the suite in. A
root a user named with `--basetemp` is left alone.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

#: Set by mutmut in every process it runs the suite in ("stats", "", or a mutant name).
MUTMUT_ENV = "MUTANT_UNDER_TEST"


def session_root(config: pytest.Config) -> Path | None:
    """The numbered root this session made, or None if it made none."""
    if config.option.basetemp:
        return None  # the user's directory, not pytest's numbered one
    factory = getattr(config, "_tmp_path_factory", None)
    root = getattr(factory, "_basetemp", None)
    return root if isinstance(root, Path) else None


def pytest_sessionfinish(session: pytest.Session) -> None:
    if MUTMUT_ENV not in os.environ:
        return
    root = session_root(session.config)
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)
