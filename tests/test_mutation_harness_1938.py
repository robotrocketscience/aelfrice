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
import shlex
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


#: Shell control operators that end one command and start the next.
_SHELL_OPERATORS = frozenset({"&&", "||", ";", "|", "&"})


def _contract_step(workflow: str) -> list[str]:
    """The contract step's lines, from its `- name:` line to the next step."""
    step = workflow[workflow.index("- name: Check the mutmut contract the plan relies on"):]
    return step[:step.index("- name: Plan the shards")].splitlines()


def _step_env(lines: list[str]) -> set[str]:
    """Each `KEY: value` line of the step, with YAML comments removed."""
    return {
        line.split(" #", 1)[0].strip()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    }


def _pytest_args(lines: list[str]) -> list[str]:
    """The arguments of every `pytest` command in the step's `run:` script.

    The script is tokenised as the shell reads it: a backslash-newline
    joins two lines, `#` starts a comment, and a control operator ends
    the command. So a test named only in a comment, or by another
    command such as `echo`, is not an argument.
    """
    start = next(i for i, line in enumerate(lines) if line.strip() == "run: |")
    script = "\n".join(lines[start + 1:]).replace("\\\n", " ")
    args: list[str] = []
    for line in script.splitlines():
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        in_pytest = False
        for token in lexer:
            if token in _SHELL_OPERATORS:
                in_pytest = False
            elif in_pytest:
                args.append(token)
            elif token == "pytest":
                in_pytest = True
    return args


def test_the_contract_step_runs_this_test() -> None:
    """It only guards the weekly run if the plan job runs it with mutmut."""
    lines = _contract_step(_workflow())
    assert 'AELF_REQUIRE_MUTMUT: "1"' in _step_env(lines)
    assert f"tests/{Path(__file__).name}" in _pytest_args(lines)


@pytest.mark.parametrize(
    "replacement",
    [
        "\n          # tests/test_mutation_harness_1938.py runs later.\n",
        "  # tests/test_mutation_harness_1938.py\n",
        "\n          echo tests/test_mutation_harness_1938.py\n",
        " && echo tests/test_mutation_harness_1938.py\n",
    ],
    ids=["comment-line", "trailing-comment", "other-command", "after-operator"],
)
def test_a_mention_outside_the_pytest_command_does_not_count(
    replacement: str,
) -> None:
    """The step reader must not find this test where pytest won't run it."""
    workflow = _workflow()
    command = (
        "tests/test_mutation_function_scope_1632.py \\\n"
        "              tests/test_mutation_harness_1938.py\n"
    )
    assert workflow.count(command) == 1
    edited = workflow.replace(
        command, "tests/test_mutation_function_scope_1632.py" + replacement,
    )
    args = _pytest_args(_contract_step(edited))
    assert "tests/test_mutation_function_scope_1632.py" in args
    assert f"tests/{Path(__file__).name}" not in args


def test_a_commented_out_env_line_does_not_count() -> None:
    lines = _contract_step(_workflow().replace(
        '          AELF_REQUIRE_MUTMUT: "1"\n',
        '          # AELF_REQUIRE_MUTMUT: "1"\n',
    ))
    assert 'AELF_REQUIRE_MUTMUT: "1"' not in _step_env(lines)


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
