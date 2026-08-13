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
that imports the real hook. They assert behaviour — a test that sleeps
past its own budget and survives — rather than inspecting the marker,
because the marker is only interesting if pytest-timeout honours it.
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

# A workflow "invokes pytest" when it has a run step that executes it.
# Prose mentions do not count: four workflows name pytest in a comment
# and run nothing (`staging-gate`, `replay-soak-gate`, `pr-metadata`,
# `post-release-docs-issue`), and pinning those would assert a fiction.
_INVOKES_PYTEST = re.compile(r"^\s*(?:-\s*run:|run:)?.*\b(?:uv run|python -m)\s+pytest\b")

_SUBPROCESS_BUDGET_S = 60


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
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "",
    }
    if scale is not None:
        env[_PIN] = scale
    return subprocess.run(
        [
            sys.executable, "-m", "pytest",
            str(tmp_path / "test_probe.py"),
            "-q", "-p", "no:cacheprovider",
            "-o", f"timeout={ini_timeout}",
        ],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        timeout=_SUBPROCESS_BUDGET_S,
    )


_MARKED = (
    "import time\n"
    "import pytest\n"
    "@pytest.mark.timeout(2)\n"
    "def test_probe():\n"
    "    time.sleep(3)\n"
)

_UNMARKED = (
    "import time\n"
    "def test_probe():\n"
    "    time.sleep(3)\n"
)


def test_the_scale_reaches_a_test_that_carries_its_own_marker(
    tmp_path: Path,
) -> None:
    """The load-bearing case. A marker of 2 s must become 8 s at 4x.

    This is the assertion that dies if `append=False` becomes
    `append=True`: the injected marker then sits behind the test's own,
    `get_closest_marker` returns 2, and the sleep is killed at 2.0 s
    while every unmarked test still passes. A guard that exercises only
    the unmarked path passes on both trees and pins nothing.
    """
    result = _run(tmp_path, _MARKED, ini_timeout=1, scale="4")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Timeout" not in result.stdout


def test_the_scale_reaches_a_test_with_no_marker(tmp_path: Path) -> None:
    """The ini base is scaled too, not only the markers."""
    result = _run(tmp_path, _UNMARKED, ini_timeout=1, scale="4")
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_scale_of_one_restores_the_exact_unscaled_budget(
    tmp_path: Path,
) -> None:
    """CI's pin must reproduce today's behaviour to the tenth of a second.

    The reported budget is asserted, not merely the failure: a scale of
    1 that still passed through the multiply would fail here too, but at
    a different number, and "it failed" would not tell them apart.
    """
    result = _run(tmp_path, _MARKED, ini_timeout=1, scale="1")
    assert result.returncode != 0
    assert "Timeout (>2.0s)" in result.stdout, result.stdout


def test_an_unset_scale_still_gives_headroom(tmp_path: Path) -> None:
    """The default is the developer default, so an unset var is not 1x."""
    result = _run(tmp_path, _MARKED, ini_timeout=1, scale=None)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_malformed_scale_falls_back_rather_than_raising(
    tmp_path: Path,
) -> None:
    """A typo in the env var must not decide how long the suite may run.

    Zero is the case that matters: `pytest-timeout` reads a timeout of 0
    as *disabled*, so a scale that multiplied through to 0 would silently
    remove every budget in the suite — the one outcome worse than a
    budget that is too tight.
    """
    for bad in ("nonsense", "0", "-2", ""):
        result = _run(tmp_path, _MARKED, ini_timeout=1, scale=bad)
        assert result.returncode == 0, f"scale={bad!r}: {result.stdout}"


def _pytest_workflows() -> list[Path]:
    return sorted(
        p for p in _WORKFLOWS.glob("*.yml")
        if any(_INVOKES_PYTEST.match(line) for line in p.read_text().splitlines())
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
    assert found, "no workflow appears to invoke pytest — the regex has rotted"
    missing = [p.name for p in found if _PIN not in p.read_text()]
    assert not missing, (
        f"these workflows run pytest without pinning {_PIN}: {missing}. "
        f"CI must not inherit the developer scale."
    )


def test_the_pin_is_the_unscaled_value() -> None:
    """Pinning it to something other than 1 would be worse than not pinning."""
    for path in _pytest_workflows():
        text = path.read_text()
        assert re.search(rf'{_PIN}:\s*"?1"?\s*$', text, re.M), (
            f"{path.name} sets {_PIN} to something other than 1"
        )


def test_the_base_timeout_leaves_room_for_the_slowest_unit_test() -> None:
    """The ini floor, pinned by value.

    An assertion that only read the key would pass at the old 5 s, which
    is the value the issue was filed against.
    """
    with (_REPO / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)
    assert config["tool"]["pytest"]["ini_options"]["timeout"] == 30
