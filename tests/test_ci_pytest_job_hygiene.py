"""The `pytest` job in ci.yml keeps its diagnostics and its required names.

The job carries the two required checks `pytest (3.12)` and `pytest (3.13)`.
These tests pin the keys that make a red or hung leg diagnosable without a
re-run: a pinned runner image, `fail-fast: false`, a job timeout, an
interpreter log, the pytest summary and JUnit flags, and an upload of the
JUnit file that runs on failure. They also pin the check names, because the
branch ruleset and `scripts/merge_train_gate.py` match them as strings and a
job-level `name:` or a matrix change would rename both.

The job deselects one test, the live-tree producer check, because the
`derived-figures` job runs the same producers on every PR. The tests at the
bottom pin the legs' test selection and that job's gate command verbatim,
and reject the keys that skip the job or let it fail green.

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

# The one test the required legs skip. `derived-figures` runs its producers.
_DESELECTED = (
    "tests/test_derived_figures_1469.py::test_the_repo_passes_the_producer_checks"
)

# The pytest command's test selection, in order. Every other token must be a
# report flag, which changes what the log shows and not what runs.
_SELECTION = ["tests/", "--ignore=tests/e2e", "--deselect", _DESELECTED]
_REPORT_FLAGS = {"-q", "-rfEsxX", "--durations=25"}

# The `derived-figures` gate command, verbatim. A path argument replaces the
# scanned roots, `--restamp` or `--list-unmarked` returns before the producers
# run, and `|| true` turns a hard failure into a pass.
_GATE_RUNS = {
    f"uv run python scripts/check_derived_figures.py --mode {mode} --github"
    for mode in ("all", "producers")
}

# The job-level keys `derived-figures` may carry. `if`, `needs` and `strategy`
# can skip the job, and the merge train does not count `skipped` as failing.
# `continue-on-error` turns a red job green, and `name` renames the check.
_DERIVED_FIGURES_KEYS = {"runs-on", "steps", "timeout-minutes", "permissions"}


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _job_lines(job: str = "pytest") -> list[str]:
    """The body of `jobs.<job>`, whole-line comments and blanks dropped."""
    lines = _CI.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"  {job}:")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if _indent(line) <= 2:
            break
        body.append(line)
    assert body, f"found `{job}:` under jobs but no body; the parser has rotted"
    return body


def _job_keys(job: str = "pytest") -> dict[str, str]:
    """Job-level scalar keys, at four spaces."""
    out: dict[str, str] = {}
    for line in _job_lines(job):
        m = re.fullmatch(r"    ([A-Za-z-]+):\s*(.*?)\s*", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _block(key: str, at_indent: int, job: str = "pytest") -> list[str]:
    lines = _job_lines(job)
    want = " " * at_indent + key + ":"
    start = next(i for i, line in enumerate(lines) if line.rstrip() == want)
    out: list[str] = []
    for line in lines[start + 1 :]:
        if _indent(line) <= at_indent:
            break
        out.append(line)
    return out


def _steps(job: str = "pytest", minimum: int = 5) -> list[list[str]]:
    """Each step of the job as its own list of lines."""
    steps: list[list[str]] = []
    for line in _block("steps", 4, job):
        if _indent(line) == 6 and line.lstrip().startswith("- "):
            steps.append([line])
        elif steps:
            steps[-1].append(line)
    assert len(steps) >= minimum, (
        f"parsed only {len(steps)} steps; the parser has rotted"
    )
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
    """The step that runs the suite, as opposed to the docs-only partial run."""
    found = [
        s for s in _steps()
        if (_field(s, "run") or "").startswith("uv run pytest tests/ ")
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


def test_no_continue_on_error_anywhere_in_the_job() -> None:
    """On the job or on a step, it turns a red suite into a green required check."""
    found = [line.strip() for line in _job_lines() if "continue-on-error:" in line]
    assert found == [], found


def test_the_legs_deselect_exactly_the_producer_check() -> None:
    """`-k`, `-m`, a second `--ignore` or `--deselect`, `--collect-only` or
    `|| true` would hide tests from the required legs with nothing else
    running them, so every token outside the report flags is pinned. A
    renamed test turns the deselect into a no-op."""
    run = (_field(_pytest_step(), "run") or "").replace(
        f"--junitxml={_junit_name()}", ""
    )
    tokens = run.split()
    assert tokens[:3] == ["uv", "run", "pytest"], tokens
    selection = [t for t in tokens[3:] if t not in _REPORT_FLAGS]
    assert selection == _SELECTION, (
        f"the pytest legs select {selection!r}, not {_SELECTION!r}"
    )
    path, _, func = _DESELECTED.partition("::")
    source = (_REPO / path).read_text(encoding="utf-8")
    assert re.search(rf"^def {re.escape(func)}\(", source, re.M), (
        f"{func} is not defined in {path}, so the deselect skips nothing"
    )


def test_the_deselected_check_runs_in_the_derived_figures_job() -> None:
    """The deselect holds only while `derived-figures` runs the producers
    over the default roots on every PR. A step-level key such as `if:`,
    `shell:` or `continue-on-error:` can skip the gate or let it fail green,
    so the gate step carries `run:` and nothing else."""
    steps = _steps("derived-figures", minimum=1)
    gate = [s for s in steps if "scripts/check_derived_figures.py" in "\n".join(s)]
    assert len(gate) == 1, f"expected one gate step in derived-figures, got {steps!r}"
    assert len(gate[0]) == 1, f"the gate step carries more than `run:`: {gate[0]!r}"
    assert _field(gate[0], "run") in _GATE_RUNS, (
        f"derived-figures runs {_field(gate[0], 'run')!r}, not one of "
        f"{sorted(_GATE_RUNS)!r}, so it may skip the producers the pytest "
        "legs deselect"
    )


def test_the_derived_figures_job_cannot_be_skipped_or_fail_green() -> None:
    """A skipped job, or one whose failure is green, leaves the producers
    unchecked on every PR with nothing red to show for it."""
    keys = set(_job_keys("derived-figures"))
    assert keys <= _DERIVED_FIGURES_KEYS, sorted(keys - _DERIVED_FIGURES_KEYS)
    found = [
        line.strip()
        for line in _job_lines("derived-figures")
        if re.match(r"\s*(?:-\s+)?(?:if|continue-on-error):", line)
    ]
    assert found == [], found
    assert "derived-figures" in FLOOR_NAMES
