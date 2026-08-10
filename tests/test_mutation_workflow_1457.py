"""The mutation workflow actually runs mutmut, and says when it does not (#1457).

The defect these pin: `mutation.yml` invoked mutmut with `--paths-to-mutate` and
`--tests-dir`, which mutmut 3 removed. Every scheduled run from 2026-06-21
onward died with ``Error: No such option '--paths-to-mutate'`` and reported
**success**, because both steps ended in a blanket ``|| true``. Mutation testing
had never run in CI at all — the by-hand discipline was the only mutation
coverage this project has ever had.

Two independent things have to hold, and either alone leaves the hole open:

* mutmut is *configured* — mutmut 3 reads `[tool.mutmut]` from `pyproject.toml`
  and aborts without `source_paths`. A workflow with the right command and no
  config fails exactly as loudly as the old one did, which is to say silently.
* the run's *emptiness is checked* — a step that cannot fail cannot report, and
  that is precisely what let this sit for eight weeks.

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
    text = _workflow_text()
    assert "Assert the run actually produced results" in text
    assert "::error::" in text, "a dead harness must surface as an error annotation"
    assert re.search(r"exit 1", text), "the assertion must actually fail the job"


def test_the_results_assertion_is_not_itself_swallowed() -> None:
    """`|| true` on the guard would restore the defect one layer up.

    Checked structurally: no line that greps the report for mutants may end in
    a `|| true`.
    """
    for line in _workflow_text().splitlines():
        if "__mutmut_" in line and "grep" in line:
            assert "|| true" not in line, (
                f"the mutant-presence check must be able to fail: {line.strip()}"
            )


# --- the per-PR job -----------------------------------------------------


def test_the_pr_job_is_scoped_to_the_diff() -> None:
    """AC1 — mutmut across the whole tree is a multi-hour job.

    Scoping is written into `only_mutate`, a `[tool.mutmut]` key, because
    mutmut 3 has no flag for it.
    """
    text = _workflow_text()
    assert "only_mutate" in text
    assert "git diff --name-only" in text
    assert "src/aelfrice/*.py" in text


def test_the_pr_job_cannot_block_the_merge_train() -> None:
    """AC3, and the reason it needs more than an exit code.

    merge-train gates on every non-advisory check-run on the head SHA, so an
    advisory job that reports failure blocks merges — the opposite of
    advisory. `continue-on-error` is what keeps the two consistent.
    """
    text = _workflow_text()
    assert "continue-on-error: true" in text
    assert "ADVISORY_NAMES" in text, (
        "the alternative must be named, so promoting this to blocking is a "
        "documented change rather than a rediscovery"
    )


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
