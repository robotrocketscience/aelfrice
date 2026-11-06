"""What `.github/workflows/nightly-canary.yml` has to keep doing (#1786).

The workflow can't run here, so these tests pin the lines earlier reviews found
easy to get wrong, tie the suite command to ci.yml's so the two can't drift, and
run the Python the `triage` job holds inline against small fixtures.

Parsed by line with the supply-chain test's readers: PyYAML is not in the
dependency set CI installs (#1436).
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from tests.test_workflow_supply_chain_1786 import _jobs, _permissions_block, _step_lines, _step_uses, _steps, _with_input

# The slowest test here, eight nights on a real temporary repository, took 7.3 s on 2026-11-05.
pytestmark = pytest.mark.timeout(30)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import nightly_canary as nc  # noqa: E402

_WORKFLOW = _REPO / ".github" / "workflows" / "nightly-canary.yml"
_CI = _REPO / ".github" / "workflows" / "ci.yml"
_CONTRIBUTING = _REPO / "CONTRIBUTING.md"
_STALE = _REPO / ".github" / "workflows" / "stale.yml"

_CHECKOUT = "actions/checkout@"
_MAIN_ONLY = "${{ !cancelled() && github.ref == 'refs/heads/main' }}"
# All that ci.yml's full-suite command may hold and the nightly's may not. The
# deselect leaves the producer check to ci.yml's `derived-figures` job, and this
# workflow has no such job, so the check runs in the nightly's legs.
_CI_ONLY_JUNIT = "--junitxml=junit-${{ matrix.python-version }}.xml"
_CI_ONLY_DESELECT = "--deselect tests/test_derived_figures_1469.py::test_the_repo_passes_the_producer_checks"


def _text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def _job(job: str) -> list[str]:
    return _jobs(_text())[job]


def _ci_job() -> list[str]:
    return _jobs(_CI.read_text(encoding="utf-8"))["pytest"]


def _key(job: str, key: str) -> str:
    """A job-level scalar, or a one-line list, as written."""
    for line in _job(job):
        match = re.fullmatch(rf"    {key}:\s*(.*?)\s*", line)
        if match:
            return match.group(1)
    raise AssertionError(f"{job} has no {key}")


def _step(job: str, name: str) -> str:
    found = [s for s in _steps(_job(job)) if re.search(rf"name: {re.escape(name)}\s*$", s, re.M)]
    assert len(found) == 1, (job, name, _steps(_job(job)))
    return found[0]


def _checkout(job: str) -> str:
    found = [s for s in _steps(_job(job)) if (_step_uses(s) or "").startswith(_CHECKOUT)]
    assert len(found) == 1, found
    return found[0]


def _inline_python(job: str, name: str) -> str:
    """The heredoc in a step's `run:`, as the runner's shell hands it to python3."""
    match = re.search(r"<<'PY'\n(.*?)\n\s*PY\n", _step(job, name), re.S)
    assert match, name
    return textwrap.dedent(match.group(1))


def _extras(line: str) -> list[str]:
    return re.findall(r"--extra (\S+)", line)


def _ci_sync() -> str:
    found = [line.strip() for line in _ci_job() if "uv sync" in line]
    assert len(found) == 1, found
    return found[0].partition("run: ")[2]


def _ci_suite_tokens() -> list[str]:
    found = [line for line in _ci_job() if "uv run pytest tests/ " in line and "--junitxml=" in line]
    assert len(found) == 1, found
    command = found[0].partition("run: ")[2]
    for allowed in (_CI_ONLY_JUNIT, _CI_ONLY_DESELECT):
        assert allowed in command, command
        command = command.replace(allowed, "")
    return command.split()


# --- the jobs ----------------------------------------------------------------


def test_the_jobs_and_their_needs_are_only_the_ones_that_exist() -> None:
    assert set(_jobs(_text())) == {"suite", "suite-macos", "triage", "publish", "report"}
    assert _key("triage", "needs") == "[suite, suite-macos]"
    assert _key("publish", "needs") == "[suite, suite-macos, triage]"
    assert _key("report", "needs") == "[suite-macos, publish]"


def test_the_suite_runs_the_canary_legs_and_one_report_only_leg_on_the_pinned_image() -> None:
    lines = [line.strip() for line in _job("suite")]
    assert _matrix_rows() == [("3.12", "true"), ("3.13", "true"), ("3.14", "false")]
    assert _key("suite", "name") == "suite (${{ matrix.leg }})"
    assert _key("suite", "continue-on-error") == "${{ !matrix.canary }}"
    assert "fail-fast: false" in lines
    assert _key("suite", "runs-on") == _ci_key("runs-on")


def test_the_macos_leg_is_report_only_on_its_pinned_image() -> None:
    assert _key("suite-macos", "runs-on") == "macos-15"
    assert _key("suite-macos", "continue-on-error") == "true"
    assert not [line for line in _job("suite-macos") if "matrix" in line or "strategy:" in line]


def _as_the_macos_leg(line: str) -> str:
    """A `suite` step line with that job's leg and suite cap replaced by the macOS leg's."""
    return (
        line.replace("python-version: ${{ matrix.leg }}", "python-version: '3.13'")
        .replace("name: nightly-${{ matrix.leg }}-", "name: nightly-macos-3.13-")
    )


def test_the_macos_steps_are_the_suites_but_for_the_python_the_artifact_and_one_cap() -> None:
    """So the leg stays comparable with `suite`, and a step added there comes here too."""
    suite, macos = _steps(_job("suite")), _steps(_job("suite-macos"))
    assert len(suite) == len(macos)
    capped = _step("suite-macos", "Run the suite")
    for ours, theirs in zip(suite, macos, strict=True):
        if theirs == capped:
            ours = ours.replace("timeout-minutes: 25", "timeout-minutes: 40")
        assert _as_the_macos_leg(ours) == theirs


def test_the_macos_bundle_is_one_triage_downloads_and_the_record_reads() -> None:
    upload = next(s for s in _steps(_job("suite-macos")) if "upload-artifact" in s)
    assert "if: always()" in upload
    sha = "a" * 40
    name = (_with_input(upload, "name") or "").replace("${{ github.sha }}", sha).replace("${{ github.run_attempt }}", "2")
    download = next(s for s in _steps(_job("triage")) if "download-artifact" in s)
    pattern = (_with_input(download, "pattern") or "").replace("${{ github.sha }}", sha)
    assert fnmatch.fnmatch(name, pattern), (name, pattern)
    assert nc.bundle_attempt(name, "macos-3.13", sha) == 2
    assert nc.bundle_attempt(name, "3.13", sha) is None


def _ci_key(key: str) -> str:
    return next(line.split(":", 1)[1].strip() for line in _ci_job() if line.startswith(f"    {key}:"))


def _cap(step: str) -> int | None:
    """The step's own `timeout-minutes`, or None."""
    key_indent, lines = _step_lines(step)
    for indent, text in lines:
        key, _, value = text.partition(":")
        if indent == key_indent and key == "timeout-minutes":
            return int(value)
    return None


@pytest.mark.parametrize("job", ["suite", "suite-macos", "triage", "publish", "report"])
def test_every_step_has_a_cap_and_the_caps_fit_under_the_job(job: str) -> None:
    """A job past its cap ends `cancelled`, and so does the run, which hides whether
    the night was red, and the job's bundle is lost. The jobs after it still run, as
    on run 42176049962. A step past its cap ends `failure` instead."""
    steps = _steps(_job(job))
    assert steps
    missing = [s.splitlines()[0].strip() for s in steps if _cap(s) is None]
    assert missing == []
    total = sum(_cap(s) or 0 for s in steps)
    assert total < int(_key(job, "timeout-minutes")), total


def _matrix_rows() -> list[tuple[str, str]]:
    """(leg, canary) of each row under the suite's `matrix: include:`."""
    text = "\n".join(_job("suite"))
    return re.findall(r"- leg: '([\d.]+)'\n\s+canary: (\w+)", text)


def test_only_main_reaches_each_job() -> None:
    """Under `!cancelled()` a skipped `suite` doesn't skip the jobs after it, so
    without the ref test a dispatch from a branch publishes an empty record."""
    assert _key("suite", "if") == "github.ref == 'refs/heads/main'"
    assert _key("suite-macos", "if") == _key("suite", "if")
    assert _key("triage", "if") == _MAIN_ONLY
    assert _key("publish", "if") == _MAIN_ONLY
    assert _key("report", "if") == _MAIN_ONLY


def test_a_branch_dispatch_cannot_replace_a_pending_night() -> None:
    """A group holds one pending run, so main's night needs a group of its own."""
    lines = _text().splitlines()
    start = lines.index("concurrency:")
    assert lines[start + 1] == "  group: nightly-canary-${{ github.ref }}"


def test_no_job_or_step_sets_a_pytest_variable() -> None:
    """pytest reads `PYTEST_ADDOPTS` and `PYTEST_PLUGINS`, which would change what
    runs while the pinned command stays the same."""
    found = [line for line in _text().splitlines() if "PYTEST_" in line and not line.lstrip().startswith("#")]
    assert found == []


# --- suite -------------------------------------------------------------------


def test_the_suite_command_matches_ci() -> None:
    """Same selection and report flags, but for the JUnit path and ci.yml's
    deselect, so a nightly record compares with a PR's."""
    step = _step("suite", "Run the suite")
    line = next(s.strip() for s in step.splitlines() if "uv run pytest" in s)
    command, _, tail = line.partition(" || ")
    assert tail == "rc=$?", line
    tokens = command.split()
    assert "--junitxml=bundle/junit.xml" in tokens
    assert [t for t in tokens if not t.startswith("--junitxml=")] == _ci_suite_tokens()
    assert "--deselect" not in tokens


_EXTRAS_GUARD = "The extras the tests need must be installed, or those tests skip"


def test_the_suite_syncs_what_ci_syncs() -> None:
    lines = [line.strip() for line in _job("suite")]
    assert f"- run: {_ci_sync()}" in lines


def test_the_extras_guard_covers_what_ci_syncs_and_runs_before_the_suite() -> None:
    code = _inline_python("suite", _EXTRAS_GUARD)
    loop = re.search(r"for extra in \((.*?)\):", code)
    assert loop
    assert re.findall(r'"([^"]+)"', loop.group(1)) == sorted(_extras(_ci_sync()))
    assert "uv run python - <<'PY'" in _step("suite", _EXTRAS_GUARD)
    steps = _steps(_job("suite"))
    order = [i for i, s in enumerate(steps) if _EXTRAS_GUARD in s or "name: Run the suite" in s]
    assert _EXTRAS_GUARD in steps[order[0]], order


def _guard_rc(tmp_path: Path, extras: dict[str, list[str]]) -> int:
    """The guard's exit status on a pyproject.toml holding these extras."""
    rows = "".join(f"{name} = {json.dumps(reqs)}\n" for name, reqs in extras.items())
    (tmp_path / "pyproject.toml").write_text(f"[project.optional-dependencies]\n{rows}", encoding="utf-8")
    block = tmp_path / "guard_block.py"
    block.write_text(_inline_python("suite", _EXTRAS_GUARD), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(block)], cwd=tmp_path, capture_output=True, timeout=60, check=False,
    ).returncode


_INSTALLED = ["pytest>=8.0"]
_ABSENT = ["no-such-distribution-1786>=1"]


@pytest.mark.parametrize(
    ("archive", "llm", "rc"),
    [
        (_INSTALLED, _INSTALLED, 0),
        ([*_INSTALLED, *_ABSENT], _INSTALLED, 1),
        (_INSTALLED, [*_INSTALLED, *_ABSENT], 1),
        ([], _INSTALLED, 1),
    ],
    ids=["all installed", "absent in archive", "absent in onboard-llm", "an empty extra"],
)
def test_the_extras_guard_fails_on_a_package_that_is_not_installed(
    tmp_path: Path, archive: list[str], llm: list[str], rc: int,
) -> None:
    assert _guard_rc(tmp_path, {"archive": archive, "onboard-llm": llm}) == rc


def test_the_rerun_runs_only_when_failed_ids_says_true() -> None:
    """pytest given an empty `@file` runs the whole suite."""
    rerun = _step("suite", "Rerun the failed tests once")
    assert "if: steps.failed.outputs.rerun == 'true'" in rerun
    assert "uv run pytest @bundle/failed-ids.txt -p no:cacheprovider -q --junitxml=bundle/rerun.xml" in rerun
    listing = _step("suite", "List the failed tests")
    assert "if: steps.suite.outputs.rc == '1'" in listing
    assert '--github-output "$GITHUB_OUTPUT"' in listing


def test_the_environment_is_read_under_the_legs_interpreter() -> None:
    step = _step("suite", "Record the leg's environment")
    assert "run: uv run python scripts/nightly_canary.py env --out bundle/env.json" in step


def test_the_verdict_writes_its_file_and_reads_the_rerun() -> None:
    step = _step("suite", "Sort the red tests into flaky and confirmed")
    assert "--out bundle/verdict.json" in step
    assert '--rerun bundle/rerun.xml --rerun-rc "${RERUN_RC:-124}"' in step
    assert "uv run python scripts/nightly_canary.py verdict" in step


def test_every_leg_uploads_its_bundle_even_when_it_failed() -> None:
    uploads = [s for s in _steps(_job("suite")) if "upload-artifact" in s]
    assert len(uploads) == 1, uploads
    assert "if: always()" in uploads[0]
    assert _with_input(uploads[0], "name") == "nightly-${{ matrix.leg }}-${{ github.sha }}-a${{ github.run_attempt }}"
    assert _with_input(uploads[0], "path") == "bundle/"


# --- triage ------------------------------------------------------------------


def test_triage_holds_full_history_and_no_token_in_its_checkout() -> None:
    step = _checkout("triage")
    assert _with_input(step, "fetch-depth") == "0"
    assert _with_input(step, "persist-credentials") == "false"


def test_a_green_night_with_nothing_to_cache_does_not_fail_triage() -> None:
    setup = [s for s in _steps(_job("triage")) if (_step_uses(s) or "").startswith("astral-sh/setup-uv@")]
    assert len(setup) == 1, setup
    assert _with_input(setup[0], "ignore-nothing-to-cache") == "true"


def test_the_record_gets_a_stable_date_and_the_canary_legs() -> None:
    step = _step("triage", "Build the night's record")
    assert 'actions/runs/$GITHUB_RUN_ID" --jq .created_at' in step
    assert '--created-at "$created_at"' in step
    assert "--canary-legs 3.12,3.13" in step
    assert "--leg 3.12=bundles:canary --leg 3.13=bundles:canary --leg 3.14=bundles:extra --leg macos-3.13=bundles:extra" in step


def test_the_first_night_makes_the_folder_regressions_requires() -> None:
    """`regressions` exits 2 on a missing `--last-green-dir`, and only exit 2 from
    `ls-remote` means the branch doesn't exist: a network failure isn't a first night."""
    step = _step("triage", "Compare with the last green night")
    assert 'git ls-remote --exit-code --heads origin test-results >/dev/null || rc=$?' in step
    assert re.search(r'^\s*2\)\n.*\n\s*mkdir -p "\$RUNNER_TEMP/last-green"$', step, re.M)
    assert re.search(r'^\s*\*\)\n\s*exit "\$rc"$', step, re.M)
    assert '--last-green-dir "$RUNNER_TEMP/last-green"' in step


def test_the_bisect_gets_the_legs_python_its_triage_and_ci_extras() -> None:
    step = _step("triage", "Bisect what the last green commit still passes")
    assert "TMPDIR: ${{ runner.temp }}" in step
    assert "--worktree" not in step
    assert '--triage "night/bisect-$leg.json"' in step
    assert '--python "$leg"' in step
    assert _extras(step) == _extras(_ci_sync())


def test_the_drift_check_gets_ci_extras_too() -> None:
    code = _inline_python("triage", "Check the last green commit and split what is left by leg")
    assert re.findall(r'"--extra", "([^"]+)"', code) == _extras(_ci_sync())
    assert '"--python", leg' in code


def test_triage_uploads_the_record_after_a_failed_step() -> None:
    uploads = [s for s in _steps(_job("triage")) if "upload-artifact" in s]
    assert len(uploads) == 1, uploads
    assert "if: always()" in uploads[0]
    assert _with_input(uploads[0], "overwrite") == "true"


# --- publish -----------------------------------------------------------------


def test_publish_is_hardened_and_reads_all_of_main() -> None:
    steps = _steps(_job("publish"))
    assert (_step_uses(steps[0]) or "").startswith("step-security/harden-runner@")
    step = _checkout("publish")
    assert _with_input(step, "fetch-depth") == "0"
    assert _with_input(step, "persist-credentials") == "true"


def test_publish_installs_nothing() -> None:
    """The job holds the write token, so it runs no dependency and no test."""
    text = "".join(_steps(_job("publish")))
    assert "setup-uv" not in text
    assert "uv run" not in text
    assert "contents: write" in "\n".join(_job("publish"))


def test_publish_retries_its_push_and_creates_the_branch_once() -> None:
    step = _step("publish", "Write the night to test-results")
    assert "for try in 1 2 3 4 5; do" in step
    assert "git push origin HEAD:test-results" in step
    assert 'cd "$GITHUB_WORKSPACE"' in step
    assert '--branch-dir "$results" --main-repo .' in step
    assert 'switch --orphan "new-test-results-$try"' in step
    assert re.search(r'^\s*\*\)\n\s*exit "\$rc"$', step, re.M)


def _incomplete_check(tmp_path: Path, verdict: str) -> int:
    """The exit status of the last publish step on a record with this verdict."""
    (tmp_path / "night").mkdir()
    (tmp_path / "night" / "record.json").write_text(json.dumps({"nightly": {"verdict": verdict}}), encoding="utf-8")
    step = _step("publish", "Fail the job on an incomplete night")
    body = textwrap.dedent(step.partition("run: |\n")[2])
    return subprocess.run(["bash", "-e", "-c", body], cwd=tmp_path, capture_output=True, timeout=30, check=False).returncode


@pytest.mark.parametrize(("verdict", "rc"), [("incomplete", 1), ("red", 0), ("green", 0)])
def test_only_an_incomplete_night_fails_the_last_publish_step(tmp_path: Path, verdict: str, rc: int) -> None:
    assert _incomplete_check(tmp_path, verdict) == rc


def test_the_incomplete_check_is_the_last_step_and_reads_the_real_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """After the write, so the night lands first, and the field `record` writes."""
    last = _steps(_job("publish"))[-1]
    assert "name: Fail the job on an incomplete night" in last
    assert not re.search(r"^\s*if:", last, re.M), last
    monkeypatch.syspath_prepend(_REPO / "scripts")
    import nightly_canary

    record = nightly_canary.build_nightly_record(
        {"3.12": None}, {"3.12": None}, {"3.12": None}, {"3.12": "canary"}, {"3.12": None},
        "a" * 40, 1, 1, "schedule", "2026-11-06T02:17:03Z", {"3.12": ["no bundle"]}, ["3.12"],
    )
    assert record["nightly"]["verdict"] == "incomplete"


def _key_is(step: str, key: str, value: str) -> bool:
    """Whether the step's own `key` holds `value`, not a line inside its `run:`."""
    key_indent, lines = _step_lines(step)
    return any(indent == key_indent and text == f"{key}: {value}" for indent, text in lines)


@pytest.mark.parametrize(
    ("job", "name", "minutes"),
    [
        ("suite", "Run the suite", 25),
        ("suite", "Rerun the failed tests once", 5),
        ("suite-macos", "Run the suite", 40),
        ("suite-macos", "Rerun the failed tests once", 5),
        ("triage", "Bisect what the last green commit still passes", 21),
    ],
)
def test_the_caps_that_matter_stay_where_they_are(job: str, name: str, minutes: int) -> None:
    """The bisect's cap has to cover its 1200 s budget and its 15 s margin."""
    assert _cap(_step(job, name)) == minutes


@pytest.mark.parametrize("name", ["Run the suite", "Rerun the failed tests once"])
def test_a_step_past_its_cap_still_reaches_the_verdict(name: str) -> None:
    assert _key_is(_step("suite", name), "continue-on-error", "true")


def test_a_suite_step_killed_by_its_cap_reads_as_broken() -> None:
    assert '--rc "${SUITE_RC:-124}"' in _step("suite", "Sort the red tests into flaky and confirmed")


def test_a_failed_download_still_builds_a_record() -> None:
    downloads = [s for s in _steps(_job("triage")) if "download-artifact" in s]
    assert len(downloads) == 1, downloads
    assert _key_is(downloads[0], "continue-on-error", "true")


def test_triage_can_read_the_merged_prs() -> None:
    lines = [line.strip() for line in _job("triage")]
    start = lines.index("permissions:")
    block = lines[start + 1 : lines.index("steps:")]
    assert "pull-requests: read" in block
    assert "actions: read" in block
    assert "contents: read" in block


def test_a_running_night_is_not_cancelled_by_the_next() -> None:
    lines = _text().splitlines()
    start = lines.index("concurrency:")
    assert lines[start + 2] == "  cancel-in-progress: false"


@pytest.mark.parametrize(
    "name",
    ["Check the last green commit and split what is left by leg", "Bisect what the last green commit still passes"],
)
def test_the_steps_that_run_repository_tests_hold_no_token(name: str) -> None:
    assert "GH_TOKEN" not in _step("triage", name)


def test_only_the_api_steps_hold_the_token() -> None:
    holders = [s.splitlines()[0].strip() for s in _steps(_job("triage")) if "GH_TOKEN" in s]
    assert holders == ["- name: Build the night's record", "- name: List the merge points"]
    outside = _text().partition("\njobs:\n")[0]
    assert "GH_TOKEN" not in outside
    assert not any(line.startswith("    env:") for line in _job("triage"))


def test_a_landed_push_exits_with_what_publish_said() -> None:
    """`publish` exits 1 after its writes when a pointer needs attention, so the
    night lands and the job still fails."""
    step = _step("publish", "Write the night to test-results")
    assert re.search(r'if git push origin HEAD:test-results; then\n\s*exit "\$published"\n', step)
    assert re.search(r'if git diff --cached --quiet; then\n.*\n\s*exit "\$published"\n', step)


def test_an_unreadable_input_stops_publish_before_a_commit() -> None:
    step = _step("publish", "Write the night to test-results")
    check = re.search(r'if \[ "\$published" -gt 1 \]; then\n\s*exit "\$published"\n', step)
    assert check
    assert check.start() < step.index("git add -A")


# --- report ------------------------------------------------------------------


_REPORT_STEP = "Open, update and close the night's issues"


def test_report_has_its_own_cap_and_the_pinned_image() -> None:
    assert _key("report", "runs-on") == _ci_key("runs-on")
    assert _key("report", "timeout-minutes") == "15"
    assert _cap(_step("report", _REPORT_STEP)) == 7


def test_report_can_read_the_results_branch_with_the_token_its_step_holds() -> None:
    """`issues` reads `index.jsonl` through `gh api`, so it needs `contents: read`
    and needs no fetch: the checkout keeps no credential."""
    step = _step("report", _REPORT_STEP)
    assert "GH_TOKEN: ${{ github.token }}" in step
    assert "git fetch" not in step


def test_report_holds_only_the_permissions_it_uses() -> None:
    block = _permissions_block(_job("report"), 4)
    assert block is not None
    assert [line.strip() for line in block.splitlines() if line.strip()] == ["contents: read", "issues: write"]


def test_no_other_job_and_not_the_workflow_can_write_issues() -> None:
    """The jobs that run repository tests must not get this token."""
    assert "issues" not in (_permissions_block(_text().splitlines(), 0) or "")
    for job in ("suite", "triage", "publish"):
        assert "issues:" not in (_permissions_block(_job(job), 4) or ""), job


def test_report_is_hardened_keeps_no_token_and_installs_nothing() -> None:
    steps = _steps(_job("report"))
    assert (_step_uses(steps[0]) or "").startswith("step-security/harden-runner@")
    assert _with_input(_checkout("report"), "persist-credentials") == "false"
    text = "".join(steps)
    for word in ("setup-uv", "uv sync", "uv run", "uvx", "pytest", "nightly_bisect"):
        assert word not in text, word


def test_only_the_issues_step_gets_the_token() -> None:
    holding = [s for s in _steps(_job("report")) if "GH_TOKEN" in s or "github.token" in s]
    assert holding == [_step("report", _REPORT_STEP)]
    assert "GH_TOKEN: ${{ github.token }}" in holding[0]


def test_report_reads_the_night_and_a_failed_download_still_reports() -> None:
    download = next(s for s in _steps(_job("report")) if "download-artifact" in s)
    assert "continue-on-error: true" in download
    assert _with_input(download, "name") == "nightly-triage-${{ github.sha }}"
    assert _with_input(download, "path") == "night"
    step = _step("report", _REPORT_STEP)
    assert "mkdir -p night" in step
    lines = [line.strip() for line in step.splitlines()]
    assert 'python3 scripts/nightly_canary.py issues --events night --repo "$GITHUB_REPOSITORY" --run-url "$RUN_URL"' in lines
    assert "RUN_URL: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}" in step


def test_the_run_url_the_workflow_builds_is_one_the_script_accepts() -> None:
    assert nc._RUN_URL.fullmatch("https://github.com/o/r/actions/runs/123")


def test_a_regression_that_still_fails_does_not_close_as_stale() -> None:
    """The canary closes its own issues, and only on a pass. The input's name is the
    one in `action.yml` at this pin: a misspelled input would exempt nothing."""
    steps = _steps(_jobs(_STALE.read_text(encoding="utf-8"))["stale"])
    stale = next(s for s in steps if (_step_uses(s) or "").startswith("actions/stale@"))
    assert "actions/stale@b5d41d4e1d5dceea10e7104786b73624c18a190f" in stale
    labels = _with_input(stale, "exempt-issue-labels")
    assert labels is not None
    assert nc.LABEL in labels.strip('"').split(",")


def _leg_bundle(directory: Path, outcomes: dict[str, str]) -> None:
    """A leg's bundle as the suite job leaves it, with no rerun."""
    directory.mkdir(parents=True)
    cases = "".join(
        f'<testcase classname="{test.partition("::")[0]}" name="{test.partition("::")[2]}" time="0">'
        f'{"<failure />" if outcome == "failed" else ""}</testcase>'
        for test, outcome in outcomes.items()
    )
    report = f'<testsuites><testsuite name="pytest">{cases}</testsuite></testsuites>'
    (directory / "junit.xml").write_text(report, encoding="utf-8")
    verdict = nc.classify_leg(1 if "failed" in outcomes.values() else 0, outcomes, None)
    (directory / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    (directory / "env.json").write_text("{}", encoding="utf-8")


def test_an_incomplete_night_fails_publish_and_report_still_opens_canary_broken(tmp_path: Path) -> None:
    """One leg left no usable report: its bundle has a green verdict and no junit.
    The record and triage are what `record` and `regressions` write."""
    sha = "c" * 40
    test = "tests.test_a::test_one"
    bundles = tmp_path / "bundles"
    _leg_bundle(bundles / f"nightly-3.12-{sha}-a1", {test: "passed"})
    _leg_bundle(bundles / f"nightly-3.13-{sha}-a1", {test: "passed"})
    (bundles / f"nightly-3.13-{sha}-a1" / "junit.xml").unlink()
    night = tmp_path / "night"
    assert nc.main([
        "record", "--leg", f"3.12={bundles}:canary", "--leg", f"3.13={bundles}:canary", "--sha", sha,
        "--run-id", "9", "--run-attempt", "1", "--created-at", "2026-11-06T02:17:03Z", "--out", str(night / "record.json"),
    ]) == 0
    branch = tmp_path / "branch"
    branch.mkdir()
    assert nc.main([
        "regressions", "--current", str(night / "record.json"), "--last-green-dir", str(branch),
        "--repo-root", str(tmp_path), "--out", str(night / "triage.json"),
    ]) == 0
    record = json.loads((night / "record.json").read_text(encoding="utf-8"))
    assert record["nightly"]["verdict"] == "incomplete"

    step = _step("publish", "Fail the job on an incomplete night")
    body = textwrap.dedent(step.partition("run: |\n")[2])
    done = subprocess.run(["bash", "-e", "-c", body], cwd=tmp_path, capture_output=True, timeout=30, check=False)
    assert done.returncode == 1

    # `needs` holds `publish` under `!cancelled()`: a failed `publish` doesn't skip `report`.
    assert _key("report", "needs") == "[suite-macos, publish]"
    assert _key("report", "if") == _MAIN_ONLY
    triage = json.loads((night / "triage.json").read_text(encoding="utf-8"))
    facts = {**nc.night_facts(record), "run_id": 9, "run_url": "https://github.com/o/r/actions/runs/9", "repo": "o/r"}
    actions = nc.plan_actions([], nc.night_events(record, triage), facts)
    assert [(a["action"], a["key"], a["title"]) for a in actions] == [
        ("create", "canary-broken", f"nightly: the canary produced no usable result on 3.13 ({sha[:7]})"),
    ]


def _workflow_flags() -> tuple[list[str], str, str]:
    """The `--leg` values `record` gets, and the `--canary-legs` of each step."""
    record = _step("triage", "Build the night's record")
    compare = _step("triage", "Compare with the last green night")
    legs = re.findall(r"--leg (\S+)", record)
    on_record, on_regressions = re.findall(r"--canary-legs (\S+)", f"{record}\n{compare}")
    return legs, on_record, on_regressions


def test_the_workflow_gives_each_matrix_leg_a_role_and_only_canary_legs_count() -> None:
    legs, on_record, on_regressions = _workflow_flags()
    rows = _matrix_rows()
    assert legs == [*(f"{leg}=bundles:{'canary' if canary == 'true' else 'extra'}" for leg, canary in rows), "macos-3.13=bundles:extra"]
    canary = ",".join(leg for leg, flag in rows if flag == "true")
    assert on_record == canary
    assert on_regressions == canary


_CUT_SHORT = "cut short"


class _Nights:
    """Nights run through the real `record`, `regressions` and `publish`, with the
    workflow's own flags, on a real repository that plays main."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.main = tmp_path / "main"
        self.main.mkdir()
        self.branch = tmp_path / "branch"
        self.branch.mkdir()
        _git(self.main, "init", "-q")
        (self.main / "tests").mkdir()
        (self.main / "tests" / "test_a.py").write_text("", encoding="utf-8")
        _git(self.main, "add", "-A")
        self.run_id = 100

    def night(
        self, date: str, *, on_313: str | None = "passed", on_314: str | None = "passed", on_macos: str | None = "passed",
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
        """One night; a leg given None uploaded nothing, and one given `_CUT_SHORT` uploaded
        what a job killed before its verdict holds. The record, the triage,
        `last-green.json` after publishing, and the night's commit."""
        self.run_id += 1
        _git(self.main, "commit", "-q", "--allow-empty", "-m", f"night {self.run_id}")
        sha = _git(self.main, "rev-parse", "HEAD")
        bundles = self.root / f"bundles-{self.run_id}"
        test = "tests.test_a::test_one"
        _leg_bundle(bundles / f"nightly-3.12-{sha}-a1", {test: "passed"})
        for leg, outcome in (("3.13", on_313), ("3.14", on_314), ("macos-3.13", on_macos)):
            directory = bundles / f"nightly-{leg}-{sha}-a1"
            if outcome == _CUT_SHORT:
                directory.mkdir(parents=True)
                (directory / "env.json").write_text("{}", encoding="utf-8")
            elif outcome is not None:
                _leg_bundle(directory, {test: outcome})
        legs, on_record, on_regressions = _workflow_flags()
        night = self.root / f"night-{self.run_id}"
        leg_args = [a for leg in legs for a in ("--leg", leg.replace("=bundles:", f"={bundles}:"))]
        assert nc.main([
            "record", *leg_args, "--sha", sha, "--run-id", str(self.run_id), "--run-attempt", "1",
            "--created-at", f"{date}T02:17:03Z", "--canary-legs", on_record, "--out", str(night / "record.json"),
        ]) == 0
        assert nc.main([
            "regressions", "--current", str(night / "record.json"), "--last-green-dir", str(self.branch),
            "--canary-legs", on_regressions, "--repo-root", str(self.main), "--out", str(night / "triage.json"),
        ]) == 0
        assert nc.main([
            "publish", "--record", str(night / "record.json"), "--triage", str(night / "triage.json"),
            "--branch-dir", str(self.branch), "--main-repo", str(self.main), "--main-ref", "HEAD",
        ]) == 0
        read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
        return read(night / "record.json"), read(night / "triage.json"), read(self.branch / "last-green.json"), sha

    def issues(self, record: dict[str, Any], triage: dict[str, Any]) -> list[dict[str, Any]]:
        facts = {**nc.night_facts(record), "run_id": self.run_id, "run_url": "https://github.com/o/r/actions/runs/1", "repo": "o/r"}
        return nc.plan_actions([], nc.night_events(record, triage), facts)


@pytest.mark.parametrize(
    ("leg", "flag", "outcome"),
    [
        ("3.14", "on_314", "failed"),
        ("3.14", "on_314", None),
        ("macos-3.13", "on_macos", "failed"),
        ("macos-3.13", "on_macos", None),
    ],
    ids=["3.14 red", "3.14 bundle missing", "macOS red", "macOS bundle missing"],
)
def test_a_report_only_leg_changes_nothing_about_the_night(tmp_path: Path, leg: str, flag: str, outcome: str | None) -> None:
    """With a baseline in place, so the leg's failure would be a candidate if it counted."""
    nights = _Nights(tmp_path)
    _record0, _triage0, green0, sha0 = nights.night("2026-11-06")
    assert green0["canary"]["sha"] == sha0
    assert green0["legs"][leg]["sha"] == sha0

    record, triage, green, sha = nights.night("2026-11-07", **{flag: outcome})
    assert record["nightly"]["roles"][leg] == "extra"
    assert record["nightly"]["status"][leg] == ("red" if outcome else "missing")
    assert record["nightly"]["verdict"] == "green"
    assert triage["events"] == []
    assert triage["candidates"] == []
    assert triage["new_failing"] == []
    assert triage["collection_errors"] == []
    assert triage["drift_check"] is None
    assert nights.issues(record, triage) == []
    sort_dir = tmp_path / "sort"
    sort_dir.mkdir()
    _after, outputs = _sort(sort_dir, triage, {})
    assert outputs["legs"] == ""
    assert not (sort_dir / "stub.log").exists()
    assert green["canary"]["sha"] == sha
    assert green["legs"][leg]["sha"] == sha0
    assert {other: entry["sha"] for other, entry in green["legs"].items() if other != leg} == dict.fromkeys(
        {"3.12", "3.13", "3.14", "macos-3.13"} - {leg}, sha,
    )


@pytest.mark.parametrize("on_macos", ["failed", None], ids=["macOS red", "macOS bundle missing"])
def test_a_failure_on_a_canary_leg_and_on_the_report_only_legs_is_checked_and_bisected_on_the_canary_leg_only(
    tmp_path: Path, on_macos: str | None,
) -> None:
    nights = _Nights(tmp_path)
    nights.night("2026-11-06")
    record, triage, _green, _sha = nights.night("2026-11-07", on_313="failed", on_314="failed", on_macos=on_macos)
    assert record["nightly"]["verdict"] == "red"
    assert [c["legs"] for c in triage["candidates"]] == [["3.13"]]
    assert list(triage["drift_check"]["legs"]) == ["3.13"]
    sort_dir = tmp_path / "sort"
    sort_dir.mkdir()
    _repo(sort_dir)
    triage["drift_check"]["sha"] = triage["last_green"]["sha"] = _git(sort_dir, "rev-parse", "HEAD")
    _after, outputs = _sort(sort_dir, triage, {"3.13": []})
    assert outputs["legs"] == "3.13"
    log = [json.loads(line) for line in (sort_dir / "stub.log").read_text(encoding="utf-8").splitlines()]
    assert [e["python"] for e in log] == ["3.13"]


@pytest.mark.parametrize("left", [None, _CUT_SHORT], ids=["no bundle", "bundle with no report"])
def test_a_canary_leg_cancelled_at_its_cap_makes_the_night_incomplete(tmp_path: Path, left: str | None) -> None:
    nights = _Nights(tmp_path)
    _record0, _triage0, _green0, sha0 = nights.night("2026-11-06")

    record, triage, green, _sha = nights.night("2026-11-07", on_313=left)
    assert record["nightly"]["status"]["3.13"] == "missing"
    assert record["nightly"]["verdict"] == "incomplete"
    assert [event["key"] for event in triage["events"]] == ["canary-broken"]
    assert green["canary"]["sha"] == sha0
    assert [(a["action"], a["key"]) for a in nights.issues(record, triage)] == [("create", "canary-broken")]


@pytest.mark.parametrize("left", [None, _CUT_SHORT], ids=["no bundle", "bundle with no report"])
def test_a_macos_leg_cancelled_at_its_cap_reads_as_missing_and_the_night_stays_green(tmp_path: Path, left: str | None) -> None:
    nights = _Nights(tmp_path)
    _record0, _triage0, _green0, sha0 = nights.night("2026-11-06")

    record, triage, green, sha = nights.night("2026-11-07", on_macos=left)
    assert record["nightly"]["status"]["macos-3.13"] == "missing"
    assert record["nightly"]["verdict"] == "green"
    assert triage["events"] == []
    assert nights.issues(record, triage) == []
    assert green["canary"]["sha"] == sha
    assert green["legs"]["macos-3.13"]["sha"] == sha0


def test_the_docs_leave_promoting_a_report_only_leg_to_an_operator() -> None:
    text = " ".join(_CONTRIBUTING.read_text(encoding="utf-8").split())
    assert "An operator decides whether to promote a report-only leg after 14 green nights, and nothing promotes it automatically." in text


def test_the_docs_say_a_cancelled_run_can_hold_a_green_night() -> None:
    text = " ".join(_CONTRIBUTING.read_text(encoding="utf-8").split())
    assert "A report-only job that passes its own cap makes the run read `cancelled`, even with green canary legs." in text
    assert "The later jobs still run, and the night's verdict in `index.jsonl` still tells you whether the night was red." in text
    assert "If you cancel a run by hand while the macOS job queues, `triage` doesn't run, so the canary legs that finished get no record that night." in text


def _streak_command() -> str:
    """The Python the CONTRIBUTING streak command hands python3."""
    text = _CONTRIBUTING.read_text(encoding="utf-8")
    match = re.search(r"git show origin/test-results:index\.jsonl \| python3 -c '\n(.*?)\n'\n```", text, re.S)
    assert match
    return match.group(1)


def _streak(branch: Path) -> dict[str, int]:
    done = subprocess.run(
        [sys.executable, "-c", _streak_command()], input=(branch / "index.jsonl").read_text(encoding="utf-8"),
        capture_output=True, text=True, timeout=30, check=True,
    )
    return {leg: int(count) for leg, count in (line.split() for line in done.stdout.splitlines())}


def test_the_documented_command_reads_the_streak_from_the_index(tmp_path: Path) -> None:
    nights = _Nights(tmp_path)
    nights.night("2026-11-06")
    assert _streak(nights.branch) == {"3.14": 1, "macos-3.13": 1}
    nights.night("2026-11-07")
    assert _streak(nights.branch) == {"3.14": 2, "macos-3.13": 2}
    nights.night("2026-11-07")
    assert _streak(nights.branch) == {"3.14": 2, "macos-3.13": 2}, "two runs on one date are one night"
    nights.night("2026-11-08", on_314="failed")
    assert _streak(nights.branch) == {"3.14": 0, "macos-3.13": 3}
    nights.night("2026-11-09")
    assert _streak(nights.branch) == {"3.14": 1, "macos-3.13": 4}
    nights.night("2026-11-10", on_314=None)
    assert _streak(nights.branch) == {"3.14": 0, "macos-3.13": 5}
    nights.night("2026-11-11", on_macos="failed")
    assert _streak(nights.branch) == {"3.14": 1, "macos-3.13": 0}
    nights.night("2026-11-12", on_macos=None)
    assert _streak(nights.branch) == {"3.14": 2, "macos-3.13": 0}
    lines = [json.loads(line) for line in (nights.branch / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [line["status"]["3.14"] for line in lines] == ["green", "green", "green", "red", "green", "missing", "green", "green"]
    assert [line["status"]["macos-3.13"] for line in lines] == ["green", "green", "green", "green", "green", "green", "red", "missing"]
    assert {line["verdict"] for line in lines} == {"green"}


def test_the_streak_command_reads_every_report_only_leg_the_workflow_records(tmp_path: Path) -> None:
    (tmp_path / "index.jsonl").write_text("", encoding="utf-8")
    legs, _on_record, _on_regressions = _workflow_flags()
    assert set(_streak(tmp_path)) == {leg.partition("=")[0] for leg in legs if leg.endswith(":extra")}


def test_a_line_from_before_the_leg_breaks_no_streak_read(tmp_path: Path) -> None:
    (tmp_path / "index.jsonl").write_text(json.dumps({"date": "2026-11-06", "status": {"3.13": "green"}}) + "\n", encoding="utf-8")
    assert _streak(tmp_path) == {"3.14": 0, "macos-3.13": 0}


def test_the_documents_describe_the_workflow() -> None:
    text = _CONTRIBUTING.read_text(encoding="utf-8")
    start = text.index("### The nightly canary")
    section = text[start : text.index("\n### ", start + 1)]
    for name in (
        "`suite`", "`suite-macos`", "`macos-15`", "`macos-3.13`", "`triage`", "`publish`", "`report`", "`test-results`", "02:17",
        "`last-green.json`", "`index.jsonl`", "A report-only leg opens no issue", "never moves the `canary` entry",
        "`nightly-regression`", "`recovered at <sha7> (<run url>)`", "The `stale` workflow skips this label",
        "`<!-- nightly-canary-event v1 key=<key> -->`",
    ):
        assert name in " ".join(section.split()), name
    cron = re.search(r"cron: '(\d+) (\d+) \* \* \*'", _text())
    assert cron
    assert f"{int(cron.group(2)):02d}:{int(cron.group(1)):02d} " in " ".join(section.split())


def test_the_user_docs_say_where_the_suite_runs() -> None:
    limits = " ".join((_REPO / "docs" / "user" / "LIMITATIONS.md").read_text(encoding="utf-8").split())
    install = " ".join((_REPO / "docs" / "user" / "INSTALL.md").read_text(encoding="utf-8").split())
    pyproject = re.sub(r"\n\s*# ", " ", (_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert f"on every pull request that changes code (`ci.yml` on `{_ci_key('runs-on')}`, Python 3.12 and 3.13)" in limits
    assert f"once a night (`nightly-canary.yml` on `{_key('suite-macos', 'runs-on')}`, Python 3.13), and no pull request runs it." in limits
    assert "a failure there blocks no merge and opens no issue" in limits
    assert "isn't a supported-tier promise" in limits
    assert "Linux runs the full test suite on every pull request that changes code" in install
    assert "a nightly job runs the full test suite on macOS as a report that blocks nothing" in install
    assert "Linux runs the full suite on every pull request that changes code" in pyproject
    assert "macOS runs it nightly as a report" in pyproject


# --- the Python the triage job holds ----------------------------------------


_STUB_STEP = """\
import argparse, json, os
p = argparse.ArgumentParser()
p.add_argument("subcommand")
p.add_argument("--tests")
p.add_argument("--deadline")
p.add_argument("--failed-out")
p.add_argument("--python")
p.add_argument("--extra", action="append")
args = p.parse_args()
with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as log:
    log.write(json.dumps({"python": args.python, "extra": args.extra, "tests": open(args.tests).read().split()}) + "\\n")
fails = json.loads(os.environ["STUB_FAILS"]).get(args.python, [])
open(args.failed_out, "w", encoding="utf-8").write("".join(f"{i}\\n" for i in fails))
raise SystemExit(1 if fails else 0)
"""


def _git(cwd: Path, *args: str) -> str:
    env = {
        "PATH": os.environ["PATH"], "HOME": str(cwd), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True, timeout=60,
    ).stdout.strip()


def _row(test: str, legs: list[str], **more: Any) -> dict[str, Any]:
    return {"test": test, "legs": legs, **more}


def _candidate(test: str, legs: list[str]) -> dict[str, Any]:
    return _row(test, legs, node_id=f"tests/{test}.py::t")


def _triage(green: str | None, **more: Any) -> dict[str, Any]:
    return {
        "format": 1, "sha": "b" * 40, "events": [], "candidates": [], "new_failing": [], "collection_errors": [],
        "drift_check": None, "last_green": {"sha": green} if green else None, **more,
    }


def _ask(tests: list[str], modules: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        "tests": tests, "node_ids": [f"tests/{t}.py::t" for t in tests], "unmapped": [],
        "collection_errors": list(modules),
    }


def _sort(tmp_path: Path, triage: dict[str, Any], fails: dict[str, list[str]]) -> tuple[dict[str, Any], dict[str, str]]:
    """Run the sort step's Python on `triage`; the triage after it and its outputs."""
    (tmp_path / "night").mkdir()
    (tmp_path / "night" / "triage.json").write_text(json.dumps(triage), encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "nightly_bisect.py").write_text(_STUB_STEP, encoding="utf-8")
    block = tmp_path / "sort_block.py"
    block.write_text(_inline_python("triage", "Check the last green commit and split what is left by leg"), encoding="utf-8")
    temp = tmp_path / "runner_temp"
    temp.mkdir()
    output = tmp_path / "github_output"
    r = subprocess.run(
        [sys.executable, str(block)], cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False,
        env={
            **os.environ, "RUNNER_TEMP": str(temp), "GITHUB_OUTPUT": str(output),
            "STUB_FAILS": json.dumps(fails), "STUB_LOG": str(tmp_path / "stub.log"),
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
        },
    )
    assert r.returncode == 0, r.stdout + r.stderr
    outputs = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    return json.loads((tmp_path / "night" / "triage.json").read_text(encoding="utf-8")), outputs


def _repo(tmp_path: Path) -> str:
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("*\n", encoding="utf-8")
    _git(tmp_path, "add", "-f", ".gitignore")
    _git(tmp_path, "commit", "-q", "-m", "seed")
    return _git(tmp_path, "rev-parse", "HEAD")


def _tests(rows: list[dict[str, Any]]) -> list[str]:
    return [row["test"] for row in rows]


def test_drift_leaves_the_triage_and_what_is_left_is_split_by_leg(tmp_path: Path) -> None:
    green = _repo(tmp_path)
    triage = _triage(
        green,
        candidates=[
            _candidate("both_legs", ["3.12", "3.13"]),
            _candidate("drifts", ["3.12", "3.13"]),
            _candidate("one_leg_drifts", ["3.12", "3.13"]),
            _candidate("regresses", ["3.12"]),
        ],
        new_failing=[_candidate("is_new", ["3.13"])],
        collection_errors=[_row("::tests.broken", ["3.13"], module="tests/broken.py")],
        drift_check={"sha": green, "legs": {
            "3.12": _ask(["both_legs", "drifts", "one_leg_drifts", "regresses"]),
            "3.13": _ask(["both_legs", "drifts", "one_leg_drifts"], ("tests/broken.py",)),
        }},
    )
    fails = {"3.12": ["tests/drifts.py::t", "tests/one_leg_drifts.py::t"], "3.13": ["tests/drifts.py::t"]}
    after, outputs = _sort(tmp_path, triage, fails)
    assert outputs == {"legs": "3.12 3.13", "good": green}
    assert _tests(after["candidates"]) == ["both_legs", "one_leg_drifts", "regresses"]
    assert after["candidates"][1]["legs"] == ["3.13"]
    event = next(e for e in after["events"] if e["kind"] == "drift")
    assert event["key"] == f"drift-{green[:12]}"
    assert event["tests"] == ["drifts", "one_leg_drifts"]
    assert {f["test"]: f["legs"] for f in event["failures"]} == {"drifts": ["3.12", "3.13"], "one_leg_drifts": ["3.12"]}
    on_312 = json.loads((tmp_path / "night" / "bisect-3.12.json").read_text(encoding="utf-8"))
    on_313 = json.loads((tmp_path / "night" / "bisect-3.13.json").read_text(encoding="utf-8"))
    assert _tests(on_312["candidates"]) == ["regresses"]
    assert _tests(on_313["candidates"]) == ["both_legs", "one_leg_drifts"]
    assert _tests(on_313["new_failing"]) == ["is_new"]
    assert _tests(on_313["collection_errors"]) == ["::tests.broken"]
    assert on_313["format"] == 1
    log = [json.loads(line) for line in (tmp_path / "stub.log").read_text(encoding="utf-8").splitlines()]
    assert [(e["python"], e["extra"]) for e in log] == [(leg, _extras(_ci_sync())) for leg in ("3.12", "3.13")]
    assert "tests/broken.py" in log[1]["tests"]


def test_a_mass_failure_whose_samples_all_fail_is_drift_for_the_whole_leg(tmp_path: Path) -> None:
    green = _repo(tmp_path)
    names = ["a", "b", "c"]
    triage = _triage(
        green, candidates=[_candidate(n, ["3.13"]) for n in names],
        drift_check={"sha": green, "legs": {"3.13": _ask(["a"])}},
    )
    after, outputs = _sort(tmp_path, triage, {"3.13": ["tests/a.py::t"]})
    assert after["candidates"] == []
    assert outputs["legs"] == ""
    assert next(e for e in after["events"] if e["kind"] == "drift")["tests"] == names


def test_a_first_night_has_nothing_to_check_or_bisect(tmp_path: Path) -> None:
    """A collection error with no baseline sits in that night's event."""
    triage = _triage(None, collection_errors=[_row("::tests.broken", ["3.13"], module="tests/broken.py")])
    after, outputs = _sort(tmp_path, triage, {})
    assert outputs == {"legs": "", "good": ""}
    assert after["events"] == []
    assert not (tmp_path / "stub.log").exists()


def test_a_night_on_the_last_green_commit_bisects_nothing(tmp_path: Path) -> None:
    triage = _triage("b" * 40, collection_errors=[_row("::tests.broken", ["3.13"], module="tests/broken.py")])
    _after, outputs = _sort(tmp_path, triage, {})
    assert outputs["legs"] == ""


def test_a_failed_checkout_of_the_last_green_commit_leaves_the_tests_to_bisect(tmp_path: Path) -> None:
    _repo(tmp_path)
    green = "f" * 40
    triage = _triage(
        green, candidates=[_candidate("regresses", ["3.13"])],
        drift_check={"sha": green, "legs": {"3.13": _ask(["regresses"])}},
    )
    after, outputs = _sort(tmp_path, triage, {"3.13": ["tests/regresses.py::t"]})
    assert outputs == {"legs": "3.13", "good": green}
    assert _tests(after["candidates"]) == ["regresses"]
    assert after["events"] == []
    assert not (tmp_path / "stub.log").exists()


_GOOD, _BAD = "a" * 40, "b" * 40


def _add_events(tmp_path: Path, triage: dict[str, Any], legs: str, results: dict[str, Any]) -> dict[str, Any]:
    """Run the add-events step's Python; the triage after it."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(_REPO / "scripts" / "nightly_bisect.py", tmp_path / "scripts" / "nightly_bisect.py")
    night = tmp_path / "night"
    night.mkdir()
    (night / "triage.json").write_text(json.dumps(triage), encoding="utf-8")
    for leg, result in results.items():
        (night / f"bisect-{leg}-result.json").write_text(json.dumps(result), encoding="utf-8")
    block = tmp_path / "add_events_block.py"
    block.write_text(_inline_python("triage", "Add the bisect's events to the triage"), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(block)], cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False,
        env={**os.environ, "LEGS": legs},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads((night / "triage.json").read_text(encoding="utf-8"))


def test_the_bisects_events_reach_the_triage_and_no_test_is_lost(tmp_path: Path) -> None:
    triage = _triage(_GOOD, candidates=[
        _candidate("found", ["3.13"]), _candidate("lost", ["3.13"]), _candidate("died", ["3.12"]),
    ])
    event = {"key": "pr-7", "kind": "pr", "pr": 7, "sha": _BAD, "preceding": _GOOD, "tests": ["found"]}
    result = {"complete": True, "merge_points": 3, "rounds": [{"log": "x"}], "reset_errors": [], "events": [event]}
    after = _add_events(tmp_path, triage, "3.12 3.13", {"3.13": result})
    by_key = {e["key"]: e for e in after["events"]}
    assert by_key["pr-7"]["tests"] == ["found"]
    assert by_key[f"range-{_GOOD[:12]}-{_BAD[:12]}"]["reasons"] == {"died": "no result", "lost": "no result"}
    assert after["bisect"]["3.13"]["rounds"] == [{"log": "x"}]
    assert after["bisect"]["3.12"]["complete"] is None


def test_a_sort_step_that_died_leaves_no_row_without_an_event(tmp_path: Path) -> None:
    """A dead step writes no `legs`, so nothing was bisected and every row is left."""
    triage = _triage(
        _GOOD, candidates=[_candidate("candidate", ["3.13"])], new_failing=[_candidate("is_new", ["3.12"])],
        collection_errors=[_row("::tests.broken", ["3.13"], module="tests/broken.py")],
    )
    after = _add_events(tmp_path, triage, "", {})
    assert [e["key"] for e in after["events"]] == [f"range-{_GOOD[:12]}-{_BAD[:12]}"]
    assert after["events"][0]["reasons"] == {t: "no result" for t in ("candidate", "is_new", "::tests.broken")}
    assert after["bisect"] == {}


@pytest.mark.parametrize("green", [None, _BAD], ids=["no baseline", "on the last green commit"])
def test_a_dead_sort_step_has_no_range_to_name_without_two_commits(tmp_path: Path, green: str | None) -> None:
    triage = _triage(green, collection_errors=[_row("::tests.broken", ["3.13"], module="tests/broken.py")])
    assert _add_events(tmp_path, triage, "", {})["events"] == []


def test_the_add_events_step_also_runs_after_a_dead_sort_step() -> None:
    step = _step("triage", "Add the bisect's events to the triage")
    assert "if: ${{ !cancelled() && (steps.sort.outputs.legs != '' || steps.sort.outcome == 'failure') }}" in step
