"""The timeout scale must reach the markers, and CI must not inherit it.

#1472. The suite's wall-clock budgets were sized for an idle machine, so
a loaded one lost tests that had no defect — and `aelf-pr-open.sh` runs
pytest with `-x`, so the first such loss ended the run. Twelve gate
attempts went that way.

Raising the ini `timeout`, or passing `--timeout`, cannot fix it. 179
tests carry their own `@pytest.mark.timeout`, and `pytest_timeout`
resolves the closest marker *first* and only falls back to the ini value
in its absence. So the ini value governs the minority of the suite. The
scale is applied at collection instead, by re-adding the marker, which is
the one place both populations are reachable.

The tests below run pytest in a subprocess against a throwaway conftest
that imports the real hook, and assert the budget it resolves as a
number. Asserting only that a run survived cannot see the case that
matters: a scale of 0 multiplies through to a timeout of 0, which
pytest-timeout reads as *disabled*, so every test passes and every
budget is gone. One end-to-end case still sleeps past its budget, because
a resolved number is only worth pinning if the plugin acts on it.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.timeout(120)

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO / ".github" / "workflows"

_PIN = "AELF_TEST_TIMEOUT_SCALE"

# A workflow "invokes pytest" when a `run:` step executes it, under any
# spelling. The first version of this matched the literal strings
# `uv run pytest` and `python -m pytest`, and review showed three live
# spellings walking straight past it — bare `pytest`, `uv run --frozen
# pytest`, and `uvx pytest`, the last of which the repo already uses in
# two workflows — plus every `.yaml` file, since the glob was `*.yml`.
# That is the failure this whole issue is about: a census that looked
# derived but was really a list.
#
# `mutmut` counts. It drives pytest in-process, so every budget in the
# suite applies to its runs too.
# A command "invokes pytest" when a `pytest` or `mutmut` token stands in
# command position — first, or preceded only by a runner and its flags.
#
# Tokens rather than one regex, deliberately. The regex form of this
# needed a nested quantifier for `uv run`'s flags, and CodeQL flagged it
# as exponential-backtracking on input like `--- ---`. A guard that can
# hang the security scan is not worth the compactness.
#
# `mutmut` counts: it drives pytest in-process, so every budget in the
# suite applies to its runs.
#
# Command position is what keeps `post-release-docs-issue.yml` out. That
# workflow writes an issue body inside a `run:` heredoc, and the body
# contains the prose "pytest count comment"; a substring match calls that
# an invocation and demands a pin for a workflow that runs no tests.
_TEST_RUNNERS = frozenset({"pytest", "mutmut"})
_LAUNCHERS = frozenset({
    "uv", "uvx", "run", "tool", "python", "python3", "poetry", "hatch",
    "-m", "exec", "sudo", "time", "xvfb-run",
})


def _invokes_pytest(command: str) -> bool:
    for segment in re.split(r"[;&|()]+", command):
        tokens = segment.split()
        for i, token in enumerate(tokens):
            if token not in _TEST_RUNNERS:
                continue
            if all(
                t.startswith("-") or t in _LAUNCHERS
                for t in tokens[:i]
            ):
                return True
    return False


_RUN_KEY = re.compile(r"^(?P<indent>\s*)(?:-\s+)?run:\s*(?P<inline>.*)$")

_SUBPROCESS_BUDGET_S = 60


def _strip_comments(text: str) -> str:
    """Drop whole-line YAML comments.

    Only whole-line ones: a `#` inside a shell command is not a comment
    to YAML, and this is used to decide what a step actually runs.
    """
    return "\n".join(
        line for line in text.splitlines()
        if not line.lstrip().startswith("#")
    )


def _run_commands(text: str) -> list[str]:
    """Every shell command a workflow's `run:` steps execute.

    Hand-parsed on purpose: PyYAML is not importable in CI (#1436), so a
    test that imports `yaml` passes locally and errors in the one place
    it has to work.

    Handles both `run: cmd` and a block scalar `run: |` followed by an
    indented body. Anything outside a `run:` value is ignored, which is
    what keeps a prose mention of pytest — including the one inside
    `post-release-docs-issue.yml`'s issue-body string — from counting as
    an invocation.
    """
    commands: list[str] = []
    lines = _strip_comments(text).splitlines()
    i = 0
    while i < len(lines):
        match = _RUN_KEY.match(lines[i])
        if match is None:
            i += 1
            continue
        inline = match.group("inline").strip()
        indent = len(match.group("indent"))
        i += 1
        if inline and inline not in {"|", ">", "|-", ">-", "|+", ">+"}:
            commands.append(inline)
            continue
        while i < len(lines):
            body = lines[i]
            if body.strip() and (len(body) - len(body.lstrip())) <= indent:
                break
            commands.append(body)
            i += 1
    return commands


def _conftest_body() -> str:
    """A conftest that registers the real hook and nothing else.

    Importing the function is what makes this a test of production code.
    Copying `tests/conftest.py` wholesale would drag in its autouse
    home-path fixtures, and re-implementing the hook here would pass on a
    tree where the real one is broken.
    """
    return (
        "import sys\n"
        f"sys.path.insert(0, {str(_REPO)!r})\n"
        "from tests.conftest import pytest_collection_modifyitems  # noqa: F401\n"
    )


def _run(
    tmp_path: Path,
    test_body: str,
    *,
    ini_timeout: float,
    scale: str | None,
) -> subprocess.CompletedProcess[str]:
    (tmp_path / "conftest.py").write_text(_conftest_body(), encoding="utf-8")
    (tmp_path / "test_probe.py").write_text(test_body, encoding="utf-8")
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(tmp_path),
    }
    if scale is not None:
        env[_PIN] = scale
    return subprocess.run(
        [
            sys.executable, "-m", "pytest",
            str(tmp_path / "test_probe.py"),
            # -s: the probe reports its resolved budget on stdout, which
            # pytest captures on a passing test.
            "-q", "-s", "-p", "no:cacheprovider",
            "-o", f"timeout={ini_timeout}",
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        timeout=_SUBPROCESS_BUDGET_S,
    )


# Probes report the budget pytest-timeout would actually apply, so the
# assertions can name a number. The first version asserted only "the run
# survived", and review showed that cannot see the case it existed for:
# with a scale of 0 the product is 0, pytest-timeout reads 0 as
# *disabled*, the sleeping test passes, and deleting the `scale > 0`
# guard left all eight tests green.
_REPORT = (
    "def test_probe(request):\n"
    "    m = request.node.get_closest_marker('timeout')\n"
    "    v = float(m.args[0]) if m and m.args else None\n"
    "    print(f'EFFECTIVE={v}')\n"
)

_MARKED_REPORT = "import pytest\n@pytest.mark.timeout(2)\n" + _REPORT
_UNMARKED_REPORT = _REPORT

_MARKED_SLEEP = (
    "import time\n"
    "import pytest\n"
    "@pytest.mark.timeout(2)\n"
    "def test_probe():\n"
    "    time.sleep(3)\n"
)


def _effective(result: subprocess.CompletedProcess[str]) -> str:
    for line in result.stdout.splitlines():
        if line.startswith("EFFECTIVE="):
            return line.split("=", 1)[1].strip()
    raise AssertionError(f"probe reported no budget:\n{result.stdout}")


def test_the_scale_reaches_a_test_that_carries_its_own_marker(
    tmp_path: Path,
) -> None:
    """The load-bearing case. A marker of 2 s becomes exactly 8 s at 4x.

    This is the assertion that dies if `append=False` becomes
    `append=True`: the injected marker then sits behind the test's own,
    `get_closest_marker` returns 2, and every unmarked test still passes.
    A guard that exercises only the unmarked path passes on both trees.

    Asserting 8.0 rather than "not 2.0" also pins the shipped scale. A
    default of 1.6 would satisfy an inequality and change every budget in
    the suite.
    """
    result = _run(tmp_path, _MARKED_REPORT, ini_timeout=1, scale="4")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _effective(result) == "8.0"


def test_the_scale_reaches_a_test_with_no_marker(tmp_path: Path) -> None:
    """The ini base is scaled too, not only the markers."""
    result = _run(tmp_path, _UNMARKED_REPORT, ini_timeout=1, scale="4")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _effective(result) == "4.0"


def test_the_marker_outranks_the_ini_value(tmp_path: Path) -> None:
    """Precedence pinned by value, not only by ordering.

    A marker of 2 against an ini of 10 must resolve from the marker, so
    4x gives 8 and not 40. Making `_resolved_timeout` read the ini first
    would leave the ordering tests green and change every marked budget
    in the suite by a factor of five here.
    """
    result = _run(tmp_path, _MARKED_REPORT, ini_timeout=10, scale="4")
    assert _effective(result) == "8.0"


def test_a_scale_of_one_restores_the_exact_unscaled_budget(
    tmp_path: Path,
) -> None:
    """CI's pin must reproduce today's behaviour exactly."""
    result = _run(tmp_path, _MARKED_REPORT, ini_timeout=1, scale="1")
    assert _effective(result) == "2.0"


def test_an_unset_scale_still_gives_headroom(tmp_path: Path) -> None:
    """The default is the developer default, so an unset var is not 1x."""
    result = _run(tmp_path, _MARKED_REPORT, ini_timeout=1, scale=None)
    assert _effective(result) == "8.0"


@pytest.mark.parametrize("bad", ["nonsense", "0", "-2", "", "1e400", "1e6"])
def test_a_malformed_scale_falls_back_to_the_default(
    tmp_path: Path, bad: str,
) -> None:
    """A typo in the env var must not decide how long the suite may run.

    Each case asserts the *resolved budget*, which is what makes the two
    dangerous ones visible:

    * `0` — pytest-timeout reads a timeout of 0 as disabled, so a scale
      that multiplied through would remove every budget in the suite
      while every test still passed. Only the number tells that apart
      from a correct fallback.
    * `1e400` — `float()` accepts it as `inf`, and handing `inf` to the
      marker aborts the whole session with INTERNALERROR rather than
      failing one test.
    """
    result = _run(tmp_path, _MARKED_REPORT, ini_timeout=1, scale=bad)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _effective(result) == "8.0"


def test_pytest_timeout_honours_the_injected_budget(tmp_path: Path) -> None:
    """One end-to-end case, because a marker value is only worth pinning
    if the plugin acts on it. Sleeping 3 s under a 2 s marker survives at
    4x and dies at 1x, with the reported budget naming which applied."""
    scaled = _run(tmp_path, _MARKED_SLEEP, ini_timeout=1, scale="4")
    assert scaled.returncode == 0, scaled.stdout + scaled.stderr

    unscaled = _run(tmp_path, _MARKED_SLEEP, ini_timeout=1, scale="1")
    assert unscaled.returncode != 0
    assert "Timeout (>2.0s)" in unscaled.stdout, unscaled.stdout


def _workflow_files() -> list[Path]:
    return sorted(
        set(_WORKFLOWS.glob("*.yml")) | set(_WORKFLOWS.glob("*.yaml"))
    )


def _pytest_workflows() -> list[Path]:
    return [
        p for p in _workflow_files()
        if any(
            _invokes_pytest(cmd)
            for cmd in _run_commands(p.read_text(encoding="utf-8"))
        )
    ]


def _pin_values(text: str) -> list[str]:
    """Every value assigned to the scale, comments excluded.

    A whole-file substring check is not enough, and review proved it
    twice: pasting the pin line into a *comment* satisfied it while the
    real value was 8, and a step-level `env:` override of 4 sat happily
    under a workflow-level pin of 1. Both left CI on the developer scale
    with a green guard. So every assignment is collected and every one
    has to be 1.
    """
    return re.findall(
        rf'^\s*{_PIN}:\s*"?([^"\s]+)"?\s*$',
        _strip_comments(text),
        re.M,
    )


def test_every_workflow_that_runs_pytest_pins_the_scale() -> None:
    """Derived from the workflows, not from a list written here.

    The list form is what #1472 is about: its own issue body undercounted
    the marker population, and a hand-maintained enumeration of workflows
    would let the next one that runs pytest inherit a 4x-looser budget
    unnoticed. `ci.yml` and `publish.yml` carry no `timeout-minutes`, so
    for those the scale is the only thing bounding a hang.
    """
    found = _pytest_workflows()
    assert found, "no workflow appears to invoke pytest — the parser has rotted"
    missing = [
        p.name for p in found
        if not _pin_values(p.read_text(encoding="utf-8"))
    ]
    assert not missing, (
        f"these workflows run pytest without pinning {_PIN}: {missing}. "
        f"CI must not inherit the developer scale."
    )


def test_every_assignment_of_the_pin_is_the_unscaled_value() -> None:
    """*Every* assignment, in every workflow, not just one that matches.

    Checked across all workflow files rather than only the ones that run
    pytest: a step-level override in a workflow that does not obviously
    run pytest today is a trap for the day it does.
    """
    for path in _workflow_files():
        values = _pin_values(path.read_text(encoding="utf-8"))
        wrong = [v for v in values if v != "1"]
        assert not wrong, (
            f"{path.name} assigns {_PIN}={wrong}; CI must run at the "
            f"unscaled budget. A step-level override counts."
        )


def test_the_base_timeout_leaves_room_for_the_slowest_unit_test() -> None:
    """The ini floor, pinned by value.

    An assertion that only read the key would pass at the old 5 s, which
    is the value the issue was filed against.
    """
    with (_REPO / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)
    assert config["tool"]["pytest"]["ini_options"]["timeout"] == 30
