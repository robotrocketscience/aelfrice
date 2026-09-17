"""The producer soak fails on a producer that is not a function of the tree.

`scripts/check_derived_figures.py` runs each producer once, so it cannot tell a
wrong figure from an unstable one: a producer whose output moved between runs
would red the gate on one draw and pass on the next, and whichever draw the
author took would be what got published. `scripts/soak_producer_figures.py`
is the check for that, and a soak that cannot fail is worth nothing — so both
arms are exercised here, on emitters small enough to run in a moment.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "soak_producer_figures.py"

_spec = importlib.util.spec_from_file_location("_soak", _SCRIPT)
assert _spec and _spec.loader
# `Any` for the same reason `tests/test_derived_figures_1469.py` uses it: pyright
# runs `tests/` in strict mode and an implicitly-typed module object turns every
# attribute read into an `Unknown`.
soak: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(soak)

_STABLE = "import json\nprint(json.dumps({'k': 20}))\n"
# One bit of per-run state, in the cheapest form that is genuinely not a
# function of the source: the process id.
_UNSTABLE = "import json, os\nprint(json.dumps({'k': os.getpid()}))\n"

# The `--in-process` arm calls `figures()` rather than a command line, so its
# emitters expose one. The unstable half cannot use the process id -- every run
# shares an interpreter, which is the whole point of the arm -- so it moves on
# module state instead, which is the class the subprocess arm cannot see.
_STABLE_IN_PROCESS = "def figures():\n    return {'k': 20}\n"
_UNSTABLE_IN_PROCESS = (
    "_n = 0\n"
    "def figures():\n"
    "    global _n\n"
    "    _n += 1\n"
    "    return {'k': _n}\n"
)


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway tree the soak treats as the repo root."""
    monkeypatch.setattr(soak, "REPO_ROOT", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    return tmp_path


@pytest.mark.timeout(60)
def test_a_producer_that_is_a_function_of_the_tree_passes(repo: Path) -> None:
    (repo / "benchmarks" / "p.py").write_text(_STABLE)
    assert soak.main(["benchmarks/p.py", "--runs", "3"]) == 0


@pytest.mark.timeout(60)
def test_a_producer_that_moves_between_runs_fails(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The arm that makes the passing arm mean something.

    The exit code is asserted, and so is the key it names: a soak that failed
    without saying which figure moved would leave the author re-running the
    producer by hand, which is the state this script replaces.
    """
    (repo / "benchmarks" / "p.py").write_text(_UNSTABLE)
    assert soak.main(["benchmarks/p.py", "--runs", "3"]) == 1
    err = capsys.readouterr().err
    assert "diverged" in err, err
    assert "differs in: k" in err, err


@pytest.mark.timeout(60)
def test_the_in_process_arm_passes_a_producer_that_does_not_move(
    repo: Path,
) -> None:
    (repo / "benchmarks" / "p.py").write_text(_STABLE_IN_PROCESS)
    assert soak.main(["benchmarks/p.py", "--runs", "3", "--in-process"]) == 0


@pytest.mark.timeout(60)
def test_the_in_process_arm_catches_what_the_subprocess_arm_cannot(
    repo: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A producer that moves on module state, soaked both ways.

    The subprocess arm runs each call in a cold interpreter, so a module global
    that accumulates across calls is reset before every one of its runs and it
    reports the producer stable. That is not a gap in this emitter: it is the
    class of instability the default arm is blind to by construction, which is
    why the in-process arm exists. Both halves are asserted here, because "the
    new arm passes" says nothing unless the old one fails to.
    """
    (repo / "benchmarks" / "p.py").write_text(_UNSTABLE_IN_PROCESS)
    assert soak.main(["benchmarks/p.py", "--runs", "3", "--in-process"]) == 1
    err = capsys.readouterr().err
    assert "diverged" in err, err
    assert "differs in: k" in err, err
    # ...and the same emitter, soaked the default way, is reported stable.
    (repo / "benchmarks" / "q.py").write_text(
        _UNSTABLE_IN_PROCESS + "import json\nprint(json.dumps(figures()))\n"
    )
    assert soak.main(["benchmarks/q.py", "--runs", "3"]) == 0


@pytest.mark.timeout(60)
def test_the_in_process_arm_refuses_to_run_concurrently(repo: Path) -> None:
    """`--jobs` is refused rather than ignored.

    `figures()` chdirs, so two in one interpreter would interleave their cwds
    and the soak would be measuring itself. Accepting the flag and silently
    running serially would leave an operator believing they had soaked under
    concurrency when they had not.
    """
    (repo / "benchmarks" / "p.py").write_text(_STABLE_IN_PROCESS)
    assert soak.main(
        ["benchmarks/p.py", "--runs", "3", "--in-process", "--jobs", "2"]
    ) == 2


@pytest.mark.timeout(60)
def test_a_dry_run_runs_no_producer(repo: Path) -> None:
    """`--dry-run` prints the plan. A producer that would crash proves it."""
    (repo / "benchmarks" / "p.py").write_text("raise SystemExit('ran')\n")
    assert soak.main(["benchmarks/p.py", "--dry-run"]) == 0
