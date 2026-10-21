"""The `pytest` job in ci.yml keeps its diagnostics and its required names.

The job carries the two required checks `pytest (3.12)` and `pytest (3.13)`.
These tests pin the keys that make a red or hung leg diagnosable without a
re-run: a pinned runner image, `fail-fast: false`, a job timeout, an
interpreter log, the pytest summary and JUnit flags, and an upload of the
JUnit file that runs on failure. They also pin the check names, because the
branch ruleset and `scripts/merge_train_gate.py` match them as strings and a
job-level `name:` or a matrix change would rename both.

Parsed by line, like the neighbouring workflow tests: PyYAML is not in the
dependency set CI installs (#1436).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_CI = _REPO / ".github" / "workflows" / "ci.yml"
sys.path.insert(0, str(_REPO / "scripts"))

from merge_train_gate import FLOOR_NAMES  # noqa: E402

# The required contexts this job must produce. They live in a branch ruleset
# outside the tree, so they are stated here rather than derived.
_REQUIRED_NAMES = {"pytest (3.12)", "pytest (3.13)"}

_UPLOAD_SHA = "ea165f8d65b6e75b540449e92b4886f43607fa02"
_CHECKOUT_SHA = "34e114876b0b11c390a56381ad16ebd13914f8d5"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _job_lines() -> list[str]:
    """The body of `jobs.pytest`, whole-line comments and blanks dropped."""
    lines = _CI.read_text(encoding="utf-8").splitlines()
    start = lines.index("  pytest:")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if _indent(line) <= 2:
            break
        body.append(line)
    assert body, "found `pytest:` under jobs but no body; the parser has rotted"
    return body


def _job_keys() -> dict[str, str]:
    """Job-level scalar keys, at four spaces."""
    out: dict[str, str] = {}
    for line in _job_lines():
        m = re.fullmatch(r"    ([A-Za-z-]+):\s*(.*?)\s*", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _block(key: str, at_indent: int) -> list[str]:
    lines = _job_lines()
    want = " " * at_indent + key + ":"
    start = next(i for i, line in enumerate(lines) if line.rstrip() == want)
    out: list[str] = []
    for line in lines[start + 1 :]:
        if _indent(line) <= at_indent:
            break
        out.append(line)
    return out


def _steps() -> list[list[str]]:
    """Each step of the job as its own list of lines."""
    steps: list[list[str]] = []
    for line in _block("steps", 4):
        if _indent(line) == 6 and line.lstrip().startswith("- "):
            steps.append([line])
        elif steps:
            steps[-1].append(line)
    assert len(steps) >= 5, f"parsed only {len(steps)} steps; the parser has rotted"
    return steps


def _field(step: list[str], key: str) -> str | None:
    for line in step:
        m = re.fullmatch(rf"\s*(?:-\s+)?{re.escape(key)}:\s*(.*?)\s*", line)
        if m:
            return m.group(1)
    return None


def _matrix() -> tuple[str, list[str]]:
    """The single matrix key and its values."""
    entries = [line.strip() for line in _block("matrix", 6)]
    assert len(entries) == 1, f"expected one matrix axis, found {entries!r}"
    m = re.fullmatch(r"([A-Za-z0-9_-]+):\s*\[(.*)\]", entries[0])
    assert m, f"unparsed matrix entry {entries[0]!r}"
    values = [v.strip().strip("'\"") for v in m.group(2).split(",")]
    return m.group(1), values


def _pytest_step() -> list[str]:
    found = [
        s for s in _steps()
        if (_field(s, "run") or "").startswith("uv run pytest ")
    ]
    assert len(found) == 1, f"expected one pytest step, found {len(found)}"
    return found[0]


def _upload_step() -> list[str]:
    found = [
        s for s in _steps()
        if "actions/upload-artifact@" in (_field(s, "uses") or "")
    ]
    assert len(found) == 1, f"expected one upload-artifact step, found {len(found)}"
    return found[0]


def _junit_name() -> str:
    run = _field(_pytest_step(), "run") or ""
    # An expression holds spaces, so a bare `\S+` would stop inside it.
    m = re.search(r"--junitxml=((?:\$\{\{[^}]*\}\}|\S)+)", run)
    assert m, f"the pytest step writes no JUnit report: {run!r}"
    return m.group(1)


def test_the_job_produces_exactly_the_required_check_names() -> None:
    """A job-level `name:` or a changed matrix renames the required checks."""
    keys = _job_keys()
    assert "name" not in keys, (
        f"jobs.pytest has `name: {keys.get('name')}`, which replaces the "
        "`pytest (<version>)` check names the ruleset and merge train require"
    )
    _, values = _matrix()
    produced = {f"pytest ({v})" for v in values}
    assert produced == _REQUIRED_NAMES
    assert _REQUIRED_NAMES <= FLOOR_NAMES


def test_the_job_runs_on_the_pinned_image() -> None:
    assert _job_keys().get("runs-on") == "ubuntu-24.04"


def test_the_job_has_a_timeout() -> None:
    assert _job_keys().get("timeout-minutes") == "30"


def test_one_leg_failing_does_not_cancel_the_other() -> None:
    entries = [line.strip() for line in _block("strategy", 4)]
    assert "fail-fast: false" in entries, entries


def test_checkout_is_pinned_and_does_not_persist_credentials() -> None:
    found = [
        s for s in _steps()
        if "actions/checkout@" in (_field(s, "uses") or "")
    ]
    assert len(found) == 1
    assert _field(found[0], "uses") == f"actions/checkout@{_CHECKOUT_SHA} # v4.3.1"
    assert _field(found[0], "persist-credentials") == "false"


def test_the_environment_is_logged_before_the_suite() -> None:
    steps = _steps()
    logs = [
        i for i, s in enumerate(steps)
        if "sqlite3.sqlite_version" in "\n".join(s)
    ]
    assert len(logs) == 1, "expected one step that logs the environment"
    body = "\n".join(steps[logs[0]])
    assert "uv run python" in body
    for needle in ("sys.version", "sys.executable", "ImageOS", "ImageVersion"):
        assert needle in body, f"the environment log no longer prints {needle}"
    assert logs[0] < steps.index(_pytest_step())


def test_the_pytest_step_keeps_its_flags_and_adds_the_report_flags() -> None:
    tokens = (_field(_pytest_step(), "run") or "").split()
    for flag in ("tests/", "--ignore=tests/e2e", "-q", "-rfEsxX", "--durations=25"):
        assert flag in tokens, f"pytest step lost {flag}: {tokens!r}"


def test_the_junit_report_is_named_per_matrix_leg() -> None:
    """Two legs writing one name would leave one report per run, not two."""
    key, _ = _matrix()
    assert _junit_name() == f"junit-${{{{ matrix.{key} }}}}.xml"


def test_the_report_is_uploaded_even_when_the_suite_fails() -> None:
    step = _upload_step()
    assert _field(step, "if") == "always()"
    assert _field(step, "uses") == f"actions/upload-artifact@{_UPLOAD_SHA} # v4.6.2"
    assert _field(step, "retention-days") == "30"
    steps = _steps()
    assert steps.index(step) > steps.index(_pytest_step())


def test_the_upload_reads_the_file_pytest_writes() -> None:
    """The edge between the two steps: a renamed report uploads nothing."""
    assert _field(_upload_step(), "path") == _junit_name()


def test_the_artifact_name_is_unique_per_leg_and_commit() -> None:
    """upload-artifact v4 refuses a second artifact with the same name."""
    key, _ = _matrix()
    name = _field(_upload_step(), "name") or ""
    assert f"${{{{ matrix.{key} }}}}" in name, name
    assert "${{ github.sha }}" in name, name
