"""The `pytest` job in ci.yml keeps its diagnostics and its required names.

The job carries the two required checks `pytest (3.12)` and `pytest (3.13)`.
These tests pin the keys that make a red or hung leg diagnosable without a
re-run: a pinned runner image, `fail-fast: false`, a job timeout, an
interpreter log, the pytest summary and JUnit flags, and an upload of the
JUnit file that runs on failure. They also pin the check names, because the
branch ruleset and `scripts/merge_train_gate.py` match them as strings and a
job-level `name:` or a matrix change would rename both.

The job deselects one test, the live-tree producer check, because the
`derived-figures` job runs the same producers on every PR, on the same
Python versions as the `pytest` matrix. The branch ruleset requires the
`pytest` legs and not that job, so each leg waits for it and fails unless
every leg of it passed. The tests at the bottom pin that edge, pin that
job's matrix and runner image to the `pytest` job's, pin its timeout, pin
the legs' `run:` line, and reject the `PYTEST_` variables that add to it.
They pin that job's gate command verbatim, pin its producer cap to the
deselected test's 120 s timeout marker, and run its arguments through the
script. They reject a job-level key outside an allowlist on that job, an
`if:` or `continue-on-error:` in it, a top-level workflow key outside an
allowlist, and a `defaults:` or `shell:` anywhere in the file.

Parsed by line, like the neighbouring workflow tests: PyYAML is not in the
dependency set CI installs (#1436).
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CI = _REPO / ".github" / "workflows" / "ci.yml"
sys.path.insert(0, str(_REPO / "scripts"))

from merge_train_gate import FLOOR_NAMES  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_cdf_hygiene", _REPO / "scripts" / "check_derived_figures.py"
)
assert _spec and _spec.loader
cdf: Any = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cdf)

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
# run, and `|| true` turns a hard failure into a pass. `--producer-timeout 120`
# is the cap the deselected test's timeout marker sets.
_GATE_RUNS = {
    f"uv run python scripts/check_derived_figures.py --mode {mode} --github"
    " --producer-timeout 120"
    for mode in ("all", "producers")
}

# The cap the gate command applies to each producer, in seconds.
_PRODUCER_CAP_S = 120

# The job-level keys `derived-figures` may carry. `if` and `needs` can skip
# the job, and the merge train does not count `skipped` as failing.
# `continue-on-error` turns a red job green, and `name` renames the check.
# `strategy` is allowed, and its whole body is pinned below: an `exclude:` or
# an empty axis would drop a leg.
_DERIVED_FIGURES_KEYS = {
    "runs-on", "steps", "timeout-minutes", "permissions", "strategy",
}

# The top-level keys ci.yml may carry. `defaults` would set a shell for every
# `run:` step in the file.
_WORKFLOW_KEYS = {"name", "on", "permissions", "env", "concurrency", "jobs"}


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


def _matrix(job: str = "pytest") -> tuple[str, list[str]]:
    """The single matrix key and its values."""
    entries = [line.strip() for line in _block("matrix", 6, job)]
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


def _module_names(source: str) -> list[str]:
    """Every name the module namespace binds, which is what pytest collects
    test functions from. A `def`, `class`, import or assignment counts at top
    level and inside an `if`, `try`, `with`, `for`, `while` or `match` block,
    since each of those binds into the module. In a `match` block a
    capture, a `*` pattern and a `**` rest pattern bind a name too. The
    bodies of functions, classes and lambdas bind local names and are
    skipped. A comprehension variable counts, which errs toward a red
    test, not a hidden one."""
    names: list[str] = []
    todo: list[ast.AST] = list(ast.parse(source).body)
    while todo:
        node = todo.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
            continue
        if isinstance(node, ast.Lambda):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.extend(a.asname or a.name for a in node.names)
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.append(node.id)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
            if node.name:
                names.append(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.append(node.rest)
        todo.extend(ast.iter_child_nodes(node))
    return sorted(names)


def test_module_names_sees_a_test_bound_inside_a_block() -> None:
    """pytest collects a test bound inside an `if`, `try`, `with`, `for`,
    `while` or `match` block, so a name there that starts with the
    deselected one also drops out of the required legs. A `match` capture,
    `*` pattern or `**` rest pattern binds a name as well. A name local to
    a function does not bind."""
    source = (
        "import os as alias\n"
        "x = 1\n"
        "if True:\n"
        "    def in_if(): pass\n"
        "try:\n"
        "    def in_try(): pass\n"
        "except ValueError as in_except:\n"
        "    pass\n"
        "with open(os.devnull) as in_with:\n"
        "    class InWith: pass\n"
        "for in_for in []:\n"
        "    in_loop_body = 1\n"
        "while False:\n"
        "    def in_while(): pass\n"
        "match x:\n"
        "    case [in_case, *in_star]:\n"
        "        def in_case_body(): pass\n"
        "    case {**in_rest}:\n"
        "        pass\n"
        "def outer():\n"
        "    def local(): pass\n"
        "    local_var = 1\n"
    )
    assert _module_names(source) == sorted([
        "alias", "x", "in_if", "in_try", "in_except", "in_with", "InWith",
        "in_for", "in_loop_body", "in_while", "in_case", "in_star",
        "in_case_body", "in_rest", "outer",
    ])


def test_the_legs_deselect_exactly_the_producer_check() -> None:
    """`-k`, `-m`, a second `--ignore` or `--deselect`, `--collect-only` or
    `|| true` would hide tests from the required legs with nothing else
    running them, so every token outside the report flags is pinned. A
    renamed test turns the deselect into a no-op. pytest matches
    `--deselect` as a node ID prefix, so any other name in the file that
    starts with the deselected one drops out of the legs too."""
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
    sharing = [n for n in _module_names(source) if n.startswith(func)]
    assert sharing == [func], (
        f"{path} binds {sharing!r}, not exactly [{func!r}]. Absent, the "
        "deselect skips nothing. Any other name that starts with it drops "
        "out of the required legs with nothing else running it."
    )


def test_no_pytest_variable_is_set_anywhere_in_the_workflow() -> None:
    """pytest adds `PYTEST_ADDOPTS` to the command line, so a `-k` or a
    `--deselect` in it hides tests from the required legs while the `run:`
    line above stays pinned. It applies from a workflow, job or step `env:`
    and from an earlier step's write to `$GITHUB_ENV`, and pytest reads
    `PYTEST_PLUGINS` the same way, so no line outside a comment names one."""
    found = [
        line.strip()
        for line in _CI.read_text(encoding="utf-8").splitlines()
        if "PYTEST_" in line and not line.lstrip().startswith("#")
    ]
    assert found == [], found


# The step right after harden-runner, verbatim with comments dropped.
# `needs` reports `success`, `failure`, `cancelled` or `skipped`, and only
# `success` passes.
_NEEDS_STEP = [
    "- name: The derived-figures job must pass",
    "env:",
    "DERIVED_FIGURES: ${{ needs.derived-figures.result }}",
    'run: test "$DERIVED_FIGURES" = success',
]


def test_a_red_derived_figures_job_turns_the_required_legs_red() -> None:
    """The ruleset requires the `pytest` legs and not `derived-figures`, so
    the deselect holds only while the legs carry that job's result. Without
    `needs` a leg never sees it. Without `!cancelled()` a failed
    `derived-figures` skips the legs, and the ruleset counts a skipped check
    as a pass. An `if:` or a `shell:` on the step would let it pass on the
    docs-only branch or without testing anything. ci.yml says the step runs
    right after harden-runner, before checkout, so its position is pinned."""
    keys = _job_keys()
    assert keys.get("needs") == "derived-figures", keys
    assert keys.get("if") == "${{ !cancelled() }}", keys
    steps = _steps()
    found = [
        s for s in steps if "needs.derived-figures.result" in "\n".join(s)
    ]
    assert len(found) == 1, f"expected one step that reads the result: {found!r}"
    assert [line.strip() for line in found[0]] == _NEEDS_STEP
    assert "step-security/harden-runner@" in (_field(steps[0], "uses") or "")
    assert steps.index(found[0]) == 1, (
        "the result check is not the step right after harden-runner"
    )


def _gate_step() -> list[str]:
    steps = _steps("derived-figures", minimum=1)
    gate = [s for s in steps if "scripts/check_derived_figures.py" in "\n".join(s)]
    assert len(gate) == 1, f"expected one gate step in derived-figures, got {steps!r}"
    return gate[0]


def test_the_deselected_check_runs_in_the_derived_figures_job() -> None:
    """The deselect holds only while `derived-figures` runs the producers
    over the default roots on every PR. A step-level key such as `if:`,
    `shell:` or `continue-on-error:` can skip the gate or let it fail green,
    so the gate step carries `run:` and nothing else."""
    gate = _gate_step()
    assert len(gate) == 1, f"the gate step carries more than `run:`: {gate!r}"
    assert _field(gate, "run") in _GATE_RUNS, (
        f"derived-figures runs {_field(gate, 'run')!r}, not one of "
        f"{sorted(_GATE_RUNS)!r}, so it may skip the producers the pytest "
        "legs deselect"
    )


def test_the_gate_command_reaches_the_producers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The test above pins the gate's arguments as a string. This one runs
    them through `main`, so a mode dispatch that stops sending them to
    `check_producers` turns red here instead of making the job text-only.
    `--mode text` must not reach the stub, or the stub proves nothing."""
    markers = [SimpleNamespace(store_backed=False)]
    calls: list[object] = []
    caps: list[object] = []

    def iter_files(roots: list[str]) -> list[Path]:
        return [Path("stub.md")]

    def check_text(files: list[Path], report: object) -> list[SimpleNamespace]:
        return markers

    def check_producers(found: object, report: object, **kwargs: object) -> None:
        calls.append(found)
        caps.append(kwargs.get("timeout_s"))

    monkeypatch.setattr(cdf, "iter_files", iter_files)
    monkeypatch.setattr(cdf, "check_text", check_text)
    monkeypatch.setattr(cdf, "check_producers", check_producers)

    tokens = (_field(_gate_step(), "run") or "").split()
    argv = tokens[tokens.index("scripts/check_derived_figures.py") + 1 :]
    assert cdf.main(argv) == 0
    assert calls == [markers], f"`{' '.join(argv)}` never ran the producers"
    assert caps == [_PRODUCER_CAP_S], (
        f"`{' '.join(argv)}` ran the producers under a {caps!r} s cap"
    )

    calls.clear()
    assert cdf.main(["--mode", "text"]) == 0
    assert calls == []


def _timeout_marker(path: str, func: str) -> int:
    """The seconds in `@pytest.mark.timeout(N)` on `func` in `path`."""
    tree = ast.parse((_REPO / path).read_text(encoding="utf-8"))
    found = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == func
    ]
    assert len(found) == 1, f"{path} defines {len(found)} `{func}`"
    seconds = [
        dec.args[0].value
        for dec in found[0].decorator_list
        if isinstance(dec, ast.Call)
        and ast.unparse(dec.func) == "pytest.mark.timeout"
        and len(dec.args) == 1
        and isinstance(dec.args[0], ast.Constant)
        and isinstance(dec.args[0].value, int)
    ]
    assert len(seconds) == 1, f"`{func}` carries {len(seconds)} timeout markers"
    return seconds[0]


def test_ci_holds_each_producer_to_the_deselected_tests_cap() -> None:
    """The deselected test's timeout marker held the producers to 120 s on
    each leg. `derived-figures` runs them in its place, so its gate command
    passes the same number to `--producer-timeout`. Without the flag the
    script's 300 s default applies."""
    tokens = (_field(_gate_step(), "run") or "").split()
    flags = [i for i, t in enumerate(tokens) if t == "--producer-timeout"]
    assert len(flags) == 1, f"the gate passes --producer-timeout {len(flags)} times"
    path, _, func = _DESELECTED.partition("::")
    marker = _timeout_marker(path, func)
    assert int(tokens[flags[0] + 1]) == marker == _PRODUCER_CAP_S, (
        f"derived-figures caps each producer at {tokens[flags[0] + 1]} s, "
        f"and `{func}` is marked {marker} s"
    )


def test_the_derived_figures_job_has_no_if_or_continue_on_error() -> None:
    """A skipped job, or one whose failure is green, leaves the producers
    unchecked on every PR with nothing red to show for it. This reads the
    job's own keys and steps. The workflow-wide shell check is below."""
    keys = set(_job_keys("derived-figures"))
    assert keys <= _DERIVED_FIGURES_KEYS, sorted(keys - _DERIVED_FIGURES_KEYS)
    found = [
        line.strip()
        for line in _job_lines("derived-figures")
        if re.match(r"\s*(?:-\s+)?(?:if|continue-on-error):", line)
    ]
    assert found == [], found


def test_derived_figures_runs_on_the_pytest_matrix_versions() -> None:
    """The legs deselect the producer check on 3.12 and 3.13, so
    `derived-figures` has to run it on both. A dropped version, an
    `exclude:` or a hard-coded `python-version` on setup-uv would leave a
    version the legs no longer cover. The merge train matches each leg's
    check name as a string, so the floor has to name both."""
    key, values = _matrix("derived-figures")
    assert (key, values) == _matrix(), (
        f"derived-figures runs {key}={values!r}, not the pytest matrix "
        f"{_matrix()!r}"
    )
    strategy = [line.strip() for line in _block("strategy", 4, "derived-figures")]
    assert strategy == [
        "fail-fast: false",
        "matrix:",
        f"{key}: [{', '.join(repr(v) for v in values)}]",
    ], strategy
    setups = [
        s for s in _steps("derived-figures", minimum=1)
        if "astral-sh/setup-uv@" in (_field(s, "uses") or "")
    ]
    assert len(setups) == 1, f"expected one setup-uv step, found {len(setups)}"
    assert _field(setups[0], "python-version") == f"${{{{ matrix.{key} }}}}"
    names = {f"derived-figures ({v})" for v in values}
    assert names <= FLOOR_NAMES, f"the floor does not name {sorted(names)}"
    assert "derived-figures" not in FLOOR_NAMES, (
        "the floor names `derived-figures`, which a matrix job never reports"
    )


def test_derived_figures_runs_on_the_legs_pinned_image() -> None:
    """The required legs fail unless `derived-figures` passed, so a label
    move under that job turns them red with no commit to point at, as one
    under the legs would. It runs on the image the legs pin."""
    image = _job_keys("derived-figures").get("runs-on")
    assert image == _job_keys().get("runs-on") == "ubuntu-24.04", (
        f"derived-figures runs on {image!r}, not the legs' ubuntu-24.04"
    )


def test_derived_figures_has_a_timeout() -> None:
    """The legs wait for `derived-figures` before they start, and the
    wait does not count toward their own 30 minutes. `--producer-timeout`
    bounds each producer and not the text scan, so without a job cap a
    hang there holds both required checks pending for the platform's 360
    minutes."""
    assert _job_keys("derived-figures").get("timeout-minutes") == "10"


def test_no_default_or_step_shell_wraps_a_run_line() -> None:
    """A `shell:` template such as `true {0}` runs nothing and exits 0. Set
    under a workflow-level or job-level `defaults: run:`, it reaches every
    `run:` step that names no shell of its own, which is every step here, so
    the required legs and the producer gate would report green having run
    nothing. No line outside a comment sets `defaults:` or `shell:`, and the
    top-level keys are pinned."""
    lines = _CI.read_text(encoding="utf-8").splitlines()
    top = {
        m.group(1)
        for line in lines
        if (m := re.fullmatch(r"([A-Za-z_-]+):.*", line))
    }
    assert top == _WORKFLOW_KEYS, sorted(top ^ _WORKFLOW_KEYS)
    found = [
        line.strip()
        for line in lines
        if re.match(r"\s*(?:-\s+)?(?:defaults|shell):", line)
    ]
    assert found == [], found
