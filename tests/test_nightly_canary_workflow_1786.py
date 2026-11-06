"""What `.github/workflows/nightly-canary.yml` has to keep doing (#1786).

The workflow can't run here, so these tests pin the lines earlier reviews found
easy to get wrong, tie the suite command to ci.yml's so the two can't drift, and
run the Python the `triage` job holds inline against small fixtures.

Parsed by line with the supply-chain test's readers: PyYAML is not in the
dependency set CI installs (#1436).
"""
from __future__ import annotations

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

from tests.test_workflow_supply_chain_1786 import _jobs, _step_lines, _step_uses, _steps, _with_input

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO / ".github" / "workflows" / "nightly-canary.yml"
_CI = _REPO / ".github" / "workflows" / "ci.yml"
_CONTRIBUTING = _REPO / "CONTRIBUTING.md"

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
    assert set(_jobs(_text())) == {"suite", "triage", "publish"}
    assert _key("triage", "needs") == "suite"
    assert _key("publish", "needs") == "[suite, triage]"


def test_the_suite_runs_only_the_canary_legs_on_the_pinned_image() -> None:
    lines = [line.strip() for line in _job("suite")]
    assert "leg: ['3.12', '3.13']" in lines
    assert "fail-fast: false" in lines
    assert _key("suite", "runs-on") == _ci_key("runs-on")


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


@pytest.mark.parametrize("job", ["suite", "triage", "publish"])
def test_every_step_has_a_cap_and_the_caps_fit_under_the_job(job: str) -> None:
    """A job past its cap ends `cancelled`, and so does the run, and `triage` and
    `publish` then lose the night. A step past its cap ends `failure` instead."""
    steps = _steps(_job(job))
    assert steps
    missing = [s.splitlines()[0].strip() for s in steps if _cap(s) is None]
    assert missing == []
    total = sum(_cap(s) or 0 for s in steps)
    assert total < int(_key(job, "timeout-minutes")), total


def test_only_main_reaches_each_job() -> None:
    """Under `!cancelled()` a skipped `suite` doesn't skip the jobs after it, so
    without the ref test a dispatch from a branch publishes an empty record."""
    assert _key("suite", "if") == "github.ref == 'refs/heads/main'"
    assert _key("triage", "if") == _MAIN_ONLY
    assert _key("publish", "if") == _MAIN_ONLY


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


def test_the_suite_syncs_what_ci_syncs() -> None:
    lines = [line.strip() for line in _job("suite")]
    assert f"- run: {_ci_sync()}" in lines
    for module in ("cryptography", "anthropic"):
        assert f'run: uv run python -c "import {module}"' in lines


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


def test_the_record_gets_a_stable_date_and_the_canary_legs() -> None:
    step = _step("triage", "Build the night's record")
    assert 'actions/runs/$GITHUB_RUN_ID" --jq .created_at' in step
    assert '--created-at "$created_at"' in step
    assert "--canary-legs 3.12,3.13" in step
    assert "--leg 3.12=bundles:canary --leg 3.13=bundles:canary" in step


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


def test_the_documents_describe_the_workflow() -> None:
    text = _CONTRIBUTING.read_text(encoding="utf-8")
    start = text.index("### The nightly canary")
    section = text[start : text.index("\n### ", start + 1)]
    for name in ("`suite`", "`triage`", "`publish`", "`test-results`", "02:17", "`last-green.json`", "`index.jsonl`"):
        assert name in section, name
    cron = re.search(r"cron: '(\d+) (\d+) \* \* \*'", _text())
    assert cron
    assert f"{int(cron.group(2)):02d}:{int(cron.group(1)):02d} " in " ".join(section.split())


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


def test_the_bisects_events_reach_the_triage_and_no_test_is_lost(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    shutil.copy(_REPO / "scripts" / "nightly_bisect.py", tmp_path / "scripts" / "nightly_bisect.py")
    night = tmp_path / "night"
    night.mkdir()
    good, bad = "a" * 40, "b" * 40
    part = {"format": 1, "candidates": [_candidate("found", ["3.13"]), _candidate("lost", ["3.13"])],
            "new_failing": [], "collection_errors": []}
    (night / "triage.json").write_text(json.dumps(_triage(good)), encoding="utf-8")
    (night / "bisect-3.13.json").write_text(json.dumps(part), encoding="utf-8")
    (night / "bisect-3.12.json").write_text(json.dumps({**part, "candidates": [_candidate("died", ["3.12"])]}), encoding="utf-8")
    event = {"key": "pr-7", "kind": "pr", "pr": 7, "sha": bad, "preceding": good, "tests": ["found"]}
    (night / "bisect-3.13-result.json").write_text(
        json.dumps({"complete": True, "merge_points": 3, "rounds": [{"log": "x"}], "reset_errors": [], "events": [event]}),
        encoding="utf-8",
    )
    block = tmp_path / "merge_block.py"
    block.write_text(_inline_python("triage", "Add the bisect's events to the triage"), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(block)], cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False,
        env={**os.environ, "LEGS": "3.12 3.13", "GOOD": good, "GITHUB_SHA": bad},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    after = json.loads((night / "triage.json").read_text(encoding="utf-8"))
    by_key = {e["key"]: e for e in after["events"]}
    assert by_key["pr-7"]["tests"] == ["found"]
    assert by_key[f"range-{good[:12]}-{bad[:12]}"]["reasons"] == {"died": "no result", "lost": "no result"}
    assert after["bisect"]["3.13"]["rounds"] == [{"log": "x"}]
    assert after["bisect"]["3.12"]["complete"] is None
