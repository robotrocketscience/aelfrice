"""The pinned mutmut kills a mutant under the installed pytest (#1938).

mutmut 3.8.0 read pytest's private `Item._nodeid` in its stats pass. pytest
9.2.0 removed it, so once the lock moved to 9.2.0 every weekly shard stopped
with `AttributeError` before running a single mutant. The shard guard failed
each shard, but only after the plan and every shard's mutant generation.

This test runs `mutmut run` over a one-function project with the pytest the
test environment has, and requires its one mutant to end `killed`. The
workflow's contract step runs it with `AELF_REQUIRE_MUTMUT=1` against the
exact mutmut the shards install, so a mutmut and pytest pair that can't run a
mutant fails the plan job in seconds. Without mutmut installed it skips, the
same way as the other contract tests.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.mutation_tmp import MUTMUT_ENV
from tests.test_mutation_shards_1747 import _require_mutmut

_REPO = Path(__file__).resolve().parents[1]

_PROJECT = {
    "src/toy/__init__.py": "",
    "src/toy/calc.py": "def add(a, b):\n    return a + b\n",
    "tests/test_calc.py": (
        "from toy.calc import add\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n"
    ),
    "pyproject.toml": (
        '[project]\nname = "toy"\nversion = "0"\n\n'
        '[tool.pytest.ini_options]\npythonpath = ["src"]\n\n'
        '[tool.mutmut]\nsource_paths = ["src/toy"]\n'
    ),
}

#: `mutmut results` prints each mutant as `    <key>: <status>`.
_RESULT = re.compile(r"^ *([^ :]+__mutmut_[0-9]+): (.+)$", re.MULTILINE)


def _mutmut(project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        k: v for k, v in os.environ.items()
        if k != MUTMUT_ENV and not k.startswith("PYTEST_")
    }
    return subprocess.run(
        [sys.executable, "-m", "mutmut", *args], cwd=project, env=env,
        capture_output=True, text=True, encoding="utf-8", check=False,
        timeout=240,
    )


@pytest.mark.timeout(300)  # two mutmut runs, each with its own pytest sessions
def test_the_pinned_mutmut_kills_a_mutant_under_this_pytest(tmp_path: Path) -> None:
    if MUTMUT_ENV in os.environ:
        pytest.skip("runs mutmut itself; not inside a mutmut run")
    _require_mutmut()
    for rel, text in _PROJECT.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    run = _mutmut(tmp_path, "run", "--max-children", "1")
    log = run.stdout + run.stderr
    assert run.returncode == 0, f"mutmut run exited {run.returncode}:\n{log[-3000:]}"
    results = _mutmut(tmp_path, "results", "--all=true")
    assert results.returncode == 0, results.stdout + results.stderr
    statuses = dict(_RESULT.findall(results.stdout))
    assert statuses, f"mutmut listed no mutant:\n{results.stdout}\n{log[-3000:]}"
    assert set(statuses.values()) == {"killed"}, (
        f"every mutant of `a + b` should be killed by `add(2, 3) == 5`: {statuses}"
    )


def _workflow() -> str:
    return (_REPO / ".github/workflows/mutation.yml").read_text(encoding="utf-8")


def test_the_contract_step_runs_this_test() -> None:
    """It only guards the weekly run if the plan job runs it with mutmut."""
    workflow = _workflow()
    step = workflow[workflow.index("- name: Check the mutmut contract the plan relies on"):]
    step = step[:step.index("- name: Plan the shards")]
    assert "AELF_REQUIRE_MUTMUT: \"1\"" in step
    assert f"tests/{Path(__file__).name}" in step


def test_every_job_pins_the_mutmut_the_contract_step_checks() -> None:
    """A shard on another mutmut than the contract step runs unchecked."""
    pins = [
        pin
        for line in _workflow().splitlines()
        if not line.lstrip().startswith("#")
        for pin in re.findall(r"mutmut==([0-9][^'\" ]*)", line)
    ]
    # The contract step, the plan, the shards, and the per-PR job.
    assert len(pins) == 4, pins
    assert len(set(pins)) == 1, pins
