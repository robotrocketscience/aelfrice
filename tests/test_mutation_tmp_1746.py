"""Under mutmut, each pytest session removes its own temporary root (#1746).

mutmut's forked workers never run pytest's `atexit` lock removal, so every
per-mutant session's `pytest-N` root survived and a long run filled the disk.
`tests/mutation_tmp.py` removes the session's own root at session end, only
when `MUTANT_UNDER_TEST` is set and the root isn't a user's `--basetemp`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests import conftest
from tests import mutation_tmp

_REPO = Path(__file__).resolve().parents[1]


class _Config:
    def __init__(self, root: Path | None, basetemp: str | None = None) -> None:
        self.option: Any = type("Option", (), {"basetemp": basetemp})()
        self._tmp_path_factory: Any = type("Factory", (), {"_basetemp": root})()


class _Session:
    def __init__(self, config: _Config) -> None:
        self.config = config


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "pytest-7"
    (root / "test_x0").mkdir(parents=True)
    (root / "pytest-7.lock").write_text("", encoding="utf-8")
    return root


def test_under_mutmut_the_session_removes_its_own_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    monkeypatch.setenv(mutation_tmp.MUTMUT_ENV, "")
    mutation_tmp.pytest_sessionfinish(_Session(_Config(root)))  # type: ignore[arg-type]
    assert not root.exists()


def test_outside_mutmut_the_root_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    monkeypatch.delenv(mutation_tmp.MUTMUT_ENV, raising=False)
    mutation_tmp.pytest_sessionfinish(_Session(_Config(root)))  # type: ignore[arg-type]
    assert root.exists()


def test_a_basetemp_the_user_named_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _root(tmp_path)
    monkeypatch.setenv(mutation_tmp.MUTMUT_ENV, "stats")
    session = _Session(_Config(root, basetemp=str(root)))
    mutation_tmp.pytest_sessionfinish(session)  # type: ignore[arg-type]
    assert root.exists()


def test_the_suite_runs_the_cleanup() -> None:
    assert conftest.pytest_sessionfinish is mutation_tmp.pytest_sessionfinish


def test_uses_a_temporary_directory(tmp_path: Path) -> None:
    """The session the end-to-end test below runs: it makes a `tmp_path`."""
    (tmp_path / "f").write_text("x", encoding="utf-8")


@pytest.mark.timeout(120)
@pytest.mark.parametrize(("mutmut", "kept"), [(True, 0), (False, 1)])
def test_a_real_session_leaves_no_root_under_mutmut(
    tmp_path: Path, mutmut: bool, kept: int,
) -> None:
    """Through the real `conftest.py`, in a child pytest with its own TMPDIR."""
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    # The name mutmut 3.8.0 sets, spelled out: the module's own constant
    # can't stand in for it.
    env.pop("MUTANT_UNDER_TEST", None)
    if mutmut:
        env["MUTANT_UNDER_TEST"] = ""
    node = f"{Path(__file__).relative_to(_REPO)}::test_uses_a_temporary_directory"
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", node],
        cwd=_REPO, env=env, capture_output=True, text=True, timeout=110, check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    # `pytest-current` is pytest's symlink to the newest root, not a root.
    roots = [r for r in tmp_path.glob("pytest-of-*/pytest-*") if not r.is_symlink()]
    assert len(roots) == kept, roots
