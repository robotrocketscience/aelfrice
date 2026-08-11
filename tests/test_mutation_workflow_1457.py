"""The mutation workflow actually runs mutmut, and says when it does not (#1457).

The defect these pin: `mutation.yml` invoked mutmut with `--paths-to-mutate` and
`--tests-dir`, which mutmut 3 removed. Every scheduled run from 2026-06-21
onward died with ``Error: No such option '--paths-to-mutate'`` and reported
**success**, because both steps ended in a blanket ``|| true``. Mutation testing
had never run in CI at all — the by-hand discipline was the only mutation
coverage this project has ever had.

Three independent things have to hold, and any one alone leaves the hole open:

* mutmut is *configured* — mutmut 3 reads `[tool.mutmut]` from `pyproject.toml`
  and aborts without `source_paths`. A workflow with the right command and no
  config fails exactly as loudly as the old one did, which is to say silently.
* the suite is *runnable where mutmut runs it* — mutmut copies `source_paths`
  plus a fixed set into `mutants/` and runs pytest from there, so anything else
  the tests read from the repo root has to be in `also_copy` or it is absent.
  The first version of this fix had the config right and still executed zero
  mutants, because `import benchmarks` raised during stats collection.
* the run's *outcome is checked* — a step that cannot fail cannot report, and
  that is precisely what let this sit for eight weeks. Checked as a positive
  signal: `mutmut results` hides killed mutants unless `--all` is passed, so an
  empty report means a clean scope, not a dead harness. What means dead harness
  is mutants left `not checked`.

Parsed textually rather than with PyYAML, which is not a dependency of this
repo and is not importable in CI; a guard that needed it would not run where it
matters.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO / ".github" / "workflows" / "mutation.yml"
_PYPROJECT = _REPO / "pyproject.toml"


def _workflow_text() -> str:
    return _WORKFLOW.read_text(encoding="utf-8")


def _joined(text: str) -> str:
    """Collapse shell backslash-continuations onto one line.

    A line-local scan over this file is defeated by the very formatting it
    uses: the survivor tally was split across two lines by a `\\` and the
    `|| true` that would have failed the check landed on the second one, so
    the guard read green over the defect it existed to catch.
    """
    return text.replace("\\\n", " ")


def _step_body(name: str) -> str:
    """The `run:` block of the step titled `name`.

    Scoping matters as much as joining. Assertions over the whole workflow
    are satisfied by any occurrence anywhere — an `exit 1` in a different job
    would stand in for the one that is supposed to fail the guard.
    """
    lines = _joined(_workflow_text()).splitlines()
    marker = f"- name: {name}"
    starts = [i for i, line in enumerate(lines) if line.strip() == marker]
    assert len(starts) == 1, f"expected exactly one step named {name!r}"
    start = starts[0]
    column = len(lines[start]) - len(lines[start].lstrip(" "))
    body: list[str] = []
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip(" ")) <= column:
            break
        body.append(line)
    return "\n".join(body)


def _mutant_line_patterns() -> list[str]:
    """Every single-quoted regex in the workflow that names a mutant key.

    Extracted rather than described, so the assertion below runs the real
    pattern against real output instead of asserting that some substring is
    present. Substring assertions are exactly why an `^`-anchored pattern —
    which cannot match mutmut's four-space-indented output at all — shipped
    with eight green tests over it.
    """
    return re.findall(r"'([^'\n]*__mutmut_[^'\n]*)'", _workflow_text())


# Lines copied verbatim from a real `mutmut results --all=true` run (mutmut
# 3.7.0, `only_mutate = ["src/aelfrice/stream_encoding.py"]`, 54 mutants,
# 46 killed / 8 survived). mutmut renders every result with
# `print(f"    {k}: {status}")` (`mutmut/__main__.py`) — the four leading
# spaces are hard-coded and unconditional, which is what an `^`-anchored
# pattern cannot match.
_REAL_RESULT_LINE = "    aelfrice.stream_encoding.x__is_utf8__mutmut_2: survived"
_REAL_KILLED_LINE = "    aelfrice.stream_encoding.x__is_utf8__mutmut_1: killed"

# What the dead harness actually wrote into the report for eight weeks.
_DEAD_REPORT = (
    "Usage: mutmut run [OPTIONS] [MUTANT_NAMES]...\n"
    "Try 'mutmut run --help' for help.\n"
    "\n"
    "Error: No such option: --paths-to-mutate\n"
)


# --- the config half ----------------------------------------------------


def test_mutmut_is_configured_in_pyproject() -> None:
    """`source_paths` is mandatory; mutmut aborts rather than guessing.

    This is the half the old workflow tried to supply by CLI flag. Without it
    mutmut exits with `Please specify it by adding "source_paths=code_dir"`,
    which the blanket `|| true` then swallowed.
    """
    config = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    mutmut = config.get("tool", {}).get("mutmut")
    assert mutmut is not None, "[tool.mutmut] is where mutmut 3 reads its config"
    assert mutmut.get("source_paths"), (
        "source_paths is mandatory — mutmut aborts without it"
    )
    assert "src/aelfrice" in mutmut["source_paths"]


def test_also_copy_carries_what_the_suite_reads_from_the_repo_root() -> None:
    """The half that made the configured harness still run zero mutants.

    mutmut copies `source_paths` plus a fixed set (`tests/`, `pyproject.toml`,
    the lockfiles) into `mutants/` and runs the suite from there. Nothing else
    at the repo root exists in that tree unless `also_copy` names it, and
    `[tool.pytest.ini_options] pythonpath = ["."]` then resolves to `mutants/`
    — so `import benchmarks` raises during stats collection, pytest collects
    nothing, and every mutant ends up `not checked`.

    Derived from the suite rather than spot-checked, so a test file that
    starts reading a new root directory fails here instead of silently
    reintroducing the empty run.
    """
    config = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    also_copy = set(config["tool"]["mutmut"].get("also_copy", []))

    # mutmut's own fixed set, plus the roots `source_paths` already puts in
    # `mutants/` — neither needs naming in `also_copy`.
    fixed = {"tests", "test", "setup.cfg", "pyproject.toml", "uv.lock"}
    fixed |= {p.split("/")[0] for p in config["tool"]["mutmut"]["source_paths"]}

    needed: set[str] = set()
    for path in sorted((_REPO / "tests").glob("test_*.py")):
        source = path.read_text(encoding="utf-8")
        # `_REPO / "benchmarks"`, `_ROOT / ".github"`, and the unnamed form
        # `Path(__file__).parent.parent / ".githooks"`: any repo-root anchor
        # joined to a literal first component. Both spellings are needed —
        # scanning only the named constants missed `.githooks`, and mutmut
        # aborts stats collection on the *first* failing test, so one missing
        # directory takes the entire run to zero mutants.
        for name in re.findall(
            r'(?:_REPO[A-Z_]*|_ROOT|repo_root|REPO_ROOT|parents\[1\]'
            r'|parent\.parent)\s*/\s*"([^"/]+)"',
            source,
        ):
            needed.add(name)

    missing = sorted(n for n in needed if n not in also_copy and n not in fixed)
    assert not missing, (
        f"the suite reads these from the repo root but mutmut never copies "
        f"them into mutants/: {missing}"
    )
    assert "benchmarks" in also_copy, (
        "benchmarks is the import that killed stats collection outright"
    )


def test_mutmut_is_pinned_to_the_major_this_workflow_targets() -> None:
    """An unpinned major bump is the failure this workflow exists to fix.

    `source_paths`, `only_mutate`, `--max-children` and the `    name: status`
    report format the steps parse are all mutmut-3 shapes. mutmut 2 → 3 is
    what removed `--paths-to-mutate`; an unpinned `uv pip install mutmut`
    leaves the next major free to do the same thing again, silently, on a
    schedule nobody watches.
    """
    installs = [
        line.strip()
        for line in _command_lines()
        if "uv pip install" in line and "mutmut" in line
    ]
    assert len(installs) == 2, "both jobs install mutmut; both must be pinned"
    for line in installs:
        assert "mutmut>=3,<4" in line, f"unpinned mutmut install: {line}"


def _command_lines() -> list[str]:
    """Workflow lines that are not comments.

    Both YAML comments and the shell comments inside `run: |` blocks start with
    `#`, so one rule covers both. Scoping to commands matters: this file
    *documents* the removed flags by name, and an assertion over the raw text
    would fire on the explanation rather than on an invocation.
    """
    return [
        line for line in _workflow_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_the_removed_mutmut_2_flags_are_not_used_anywhere() -> None:
    """The exact regression, pinned as a literal.

    mutmut 3's `run` accepts only `--max-children` and mutant names. Either of
    these flags reintroduces the eight-week silent failure.
    """
    commands = "\n".join(_command_lines())
    assert "--paths-to-mutate" not in commands
    assert "--tests-dir" not in commands
    # And the flags must still be named somewhere, so the next person reading
    # this workflow learns why the invocation looks the way it does.
    assert "--paths-to-mutate" in _workflow_text()


# --- the "it actually ran" half -----------------------------------------


def test_the_weekly_job_fails_when_the_report_has_no_mutants() -> None:
    """The guard whose absence is the whole issue.

    A harness that cannot fail reports a clean tree and a broken run
    identically. The weekly job is where correctness of the harness is
    enforced, so it must assert on the report's *content*, not on an exit code
    that `|| true` has already discarded.
    """
    guard = _step_body("Assert the run actually produced results")
    assert "::error::" in guard, "a dead harness must surface as an error annotation"
    assert re.search(r"\bexit 1\b", guard), "the assertion must actually fail the job"


def test_the_guard_fails_on_not_checked_rather_than_on_an_empty_report() -> None:
    """The premise the first version had backwards.

    `mutmut results` suppresses killed mutants unless `--all` is passed, so an
    empty report is what a *fully killed* scope produces — the opposite of the
    conclusion "empty means dead harness". Health has to be asserted as a
    positive signal instead: a non-zero mutant total, and no mutant left at
    `not checked`, which is mutmut's status for a null exit code and was the
    status of every mutant in this workflow's own first run.
    """
    guard = _step_body("Assert the run actually produced results")
    assert "not checked" in guard, (
        "a run that aborted leaves mutants 'not checked' — that is the signal"
    )
    # Two independent failure arms, each with its own `exit 1`.
    assert guard.count("exit 1") == 2, (
        "an empty mutant list and a partially-executed run are different "
        "failures and must be reported as such"
    )
    for step in ("Run mutation tests", "Run mutation tests on the diff (advisory)"):
        body = _step_body(step)
        assert "mutmut results --all=true" in body, (
            f"{step!r} must ask for every mutant: without --all the killed "
            f"ones are invisible, and a bare --all is a usage error"
        )
        # `@click.option("--all", default=False)` with no `is_flag=True`, so
        # click infers a BOOL that takes an argument. A bare `--all` writes
        # `Error: Option '--all' requires an argument.` into the report — the
        # same shape of silent-usage-error this whole issue is about. Checked
        # over commands only; the comments name the bare form to explain it.
        commands = "\n".join(
            line for line in body.splitlines() if not line.lstrip().startswith("#")
        )
        assert not re.search(r"--all(?!=)", commands), (
            f"a bare `--all` is a click usage error, not a flag: {step}"
        )


def test_the_results_assertion_is_not_itself_swallowed() -> None:
    """`|| true` on the guard would restore the defect one layer up.

    Scoped to the guard step and checked after joining backslash
    continuations. The line-local, whole-file version of this test was already
    defeated on arrival: the survivor tally *did* end in `|| true`, split
    across two lines by a `\\`, and the test was green.

    Counting greps is allowed to be tolerant — `grep -c` exits 1 on zero
    matches, which under `bash -e` would abort the step before the comparison
    runs. What may never be tolerant is the failure itself.
    """
    guard = _step_body("Assert the run actually produced results")
    for line in guard.splitlines():
        if "exit 1" in line:
            assert "|| true" not in line, (
                f"the guard's failure must be able to fail: {line.strip()}"
            )


def test_the_mutant_pattern_matches_real_mutmut_output() -> None:
    """The defect eight substring assertions could not see.

    Every pattern in this workflow that names a mutant key was anchored at
    `^`, and `mutmut results` indents every line by four hard-coded spaces —
    so the weekly guard failed on a healthy tree and the PR job's survivor
    tally was wired to zero. Both were green under tests that only asserted
    the patterns were *present*.

    Run the extracted pattern against real output instead.
    """
    patterns = _mutant_line_patterns()
    assert patterns, "no mutant-key pattern found in the workflow"
    for pattern in patterns:
        for line in (_REAL_RESULT_LINE, _REAL_KILLED_LINE):
            assert re.search(pattern, line), (
                f"{pattern!r} does not match a real `mutmut results` line: "
                f"{line!r}"
            )
        # And it must not match the dead-harness report, or the guard would
        # pass over the very failure it exists to catch.
        assert not re.search(pattern, _DEAD_REPORT, re.MULTILINE), (
            f"{pattern!r} matches the usage error a dead mutmut writes"
        )


# --- the per-PR job -----------------------------------------------------


def test_the_pr_job_is_scoped_to_the_diff() -> None:
    """AC1 — mutmut across the whole tree is a multi-hour job.

    Scoping is written into `only_mutate`, a `[tool.mutmut]` key, because
    mutmut 3 has no flag for it.
    """
    scope = _step_body("Scope mutmut to the PR's changed files")
    assert "only_mutate" in scope
    assert "git diff --name-only" in scope
    assert "src/aelfrice/*.py" in scope


def test_the_pr_scope_is_the_merge_base_diff_not_a_two_tree_diff() -> None:
    """AC1 says *changed* files, and two dots does not mean that.

    `git diff BASE HEAD` is the difference between two trees, so it also lists
    every file `main` moved since the branch point — the PR would mutate code
    it never touched, and the scoping claim would be false in the direction
    that costs the most time. `BASE...HEAD` is the merge-base diff, which is
    the PR's own changes.
    """
    scope = _joined(_step_body("Scope mutmut to the PR's changed files"))
    # Comments explain the two-dot form by name; only invocations are checked.
    diff_lines = [
        line
        for line in scope.splitlines()
        if "git diff" in line and not line.lstrip().startswith("#")
    ]
    assert diff_lines, "the scope step must derive the file list from git"
    for line in diff_lines:
        assert '"${BASE_SHA}...${HEAD_SHA}"' in line, (
            f"two-dot diff includes commits this PR did not make: {line.strip()}"
        )


def test_the_pr_job_cannot_block_the_merge_train() -> None:
    """AC3, and the reason it needs more than an exit code.

    merge-train gates on every non-advisory check-run on the head SHA, so an
    advisory job that reports failure blocks merges — the opposite of
    advisory. `continue-on-error` is what keeps the two consistent.

    The *level* is the load-bearing part and is asserted, not assumed. This is
    the repo's first job-level `continue-on-error`; every other use in
    `.github/workflows/` is on a step, where it spares that step and lets the
    job still conclude `failure`. Indented two levels further in, this line
    still parses as valid YAML and silently makes the job blocking.
    """
    lines = _workflow_text().splitlines()
    job_at = next(i for i, line in enumerate(lines) if line.rstrip() == "  diff:")
    body: list[str] = []
    for line in lines[job_at + 1:]:
        if line.strip() and not line.startswith("    "):
            break
        body.append(line)
    keys = [line for line in body if line.startswith("    ") and line[4] != " "]
    assert "    continue-on-error: true" in keys, (
        "continue-on-error must be a key of the `diff` job, not of one of its "
        "steps — a step-level flag leaves the job's check-run at `failure`, "
        "which is exactly what blocks the train"
    )
    assert "ADVISORY_NAMES" in _workflow_text(), (
        "the alternative must be named, so promoting this to blocking is a "
        "documented change rather than a rediscovery"
    )


def test_the_advisory_job_is_named_for_a_reader_of_the_check_list() -> None:
    """A check-run called `diff` says nothing about what it is or is not.

    The job's whole contract is "this is advisory, a green tick is not a
    clean result". That contract is read off the check list, where the
    check-run carries the job's `name` — or, with no `name`, its bare id.
    """
    lines = _workflow_text().splitlines()
    job_at = next(i for i, line in enumerate(lines) if line.rstrip() == "  diff:")
    name_lines = [
        line.strip()
        for line in lines[job_at + 1: job_at + 6]
        if line.startswith("    name:")
    ]
    assert name_lines, "the advisory job must carry an explicit `name:`"
    assert "advisory" in name_lines[0].lower(), (
        f"the check-run name must say it is advisory: {name_lines[0]}"
    )


def test_the_pr_report_counts_survivors_apart_from_harness_statuses() -> None:
    """A survivor tally that folds in `not checked` is not a survivor tally.

    mutmut has nine statuses. Only `survived` is a coverage signal; `no
    tests`, `timeout`, `suspicious`, `skipped` and `not checked` say the
    harness did not get an answer. Summing them produces a number that looks
    like coverage and is not — which is how a run with zero mutants executed
    reported "Surviving mutants reported: 0".
    """
    report = _joined(_step_body("Report"))
    assert "': survived$'" in report or '": survived$"' in report, (
        "the survivor count must key on the survived status specifically"
    )
    for status in ("no tests", "timeout", "suspicious", "skipped", "not checked"):
        assert status in report, f"{status!r} is not reported anywhere"


def test_the_pr_job_says_a_green_tick_is_not_a_clean_result() -> None:
    """AC3's literal ask: state it in the job's own output.

    A job that always exits 0 is indistinguishable from one that found nothing
    unless it says so where the reader is looking.
    """
    text = _workflow_text()
    assert "never fails the PR" in text
    assert "GITHUB_STEP_SUMMARY" in text


def test_the_pr_trigger_is_path_filtered() -> None:
    """Keeps this out of the merge-train presence floor (#1458).

    The floor is every check emitted by a `pull_request` workflow with no
    `paths:` filter. An unfiltered mutation job would join it and then hang
    every docs-only PR on a check that never reports.
    """
    text = _workflow_text()
    pr_at = text.index("pull_request:")
    assert "paths:" in text[pr_at:pr_at + 400]
