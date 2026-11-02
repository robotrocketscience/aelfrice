"""The merge-train gate reads the required set, not every check-run (#1397).

The defect this pins: `.github/workflows/merge-train.yml` enumerated every
check-run on the head SHA and unlabelled on any failure, so an advisory bot
could block a merge while the message said `required check(s) failed`. PR #1394
sat blocked behind a `Sourcery review` finding that was verified false.

**The fix is an exclusion list, not required-only.** Gating on the required set
alone was tried and rejected in review: a PR carries ~25 check-runs against 5
required contexts, so required-only would demote **19** real gates — including
`migration-policy-check`, whose absence once left stores unopenable — to fix 2
bots. The tests below pin that non-required checks still gate; that is the
regression the narrower design would have introduced.

Both directions are asserted, because one alone is satisfied by a broken gate:
a workflow that merges everything passes "red advisory merges", and one that
merges nothing passes "red required blocks".
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from merge_train_gate import (  # noqa: E402
    ADVISORY_NAMES,
    FLOOR_NAMES,
    NONGATING_WHILE_PENDING,
    SELF_NAMES,
    evaluate,
    latest_per_name,
    base_refusal,
    main,
    required_contexts,
)

_WORKFLOW = _REPO / ".github" / "workflows" / "merge-train.yml"

REQUIRED = {
    "secrets-scan",
    "pattern-scan",
    "history-scan",
    "pytest (3.12)",
    "pytest (3.13)",
}


def _run(
    name: str, conclusion: str | None = "success",
    status: str = "completed", started_at: str = "2026-08-06T10:00:00Z",
) -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "started_at": started_at,
    }


def _all_required_green() -> list[dict[str, Any]]:
    return [_run(name) for name in sorted(REQUIRED)]


# The whole presence floor green (#1458). `_all_required_green` is no longer a
# fully-covered head: since the floor widened past the required set, the five
# required contexts alone are the *partial rollup* this issue is about. Tests
# whose subject is "an otherwise-green head" must build from this instead.
FLOOR = REQUIRED | FLOOR_NAMES


def _all_floor_green() -> list[dict[str, Any]]:
    return [_run(name) for name in sorted(FLOOR)]


def _rules(contexts: list[str]) -> list[dict[str, Any]]:
    return [
        {"type": "pull_request", "parameters": {}},
        {
            "type": "required_status_checks",
            "parameters": {
                "required_status_checks": [{"context": c} for c in contexts]
            },
        },
    ]


# --- the two directions -------------------------------------------------


def test_red_advisory_with_everything_else_green_does_not_block() -> None:
    """The #1394 case. This is the whole point of the change."""
    runs = [*_all_floor_green(), _run("Sourcery review", "failure")]
    verdict = evaluate(runs, REQUIRED)
    assert verdict["failing"] == []
    assert verdict["pending"] == []
    assert verdict["missing"] == []
    # Still reported, so it is visible rather than silently dropped.
    assert verdict["advisory_failing"] == ["Sourcery review"]


def test_a_red_NON_required_check_still_blocks() -> None:
    """The regression required-only would have introduced.

    `migration-policy-check` is not in the ruleset's required set, and it
    exists because a migration once collided the `edges` primary key and left
    stores unopenable forever. Demoting it to advisory to silence a review bot
    trades a 2-check problem for a 19-check hole.
    """
    runs = [*_all_required_green(), _run("migration-policy-check", "failure")]
    verdict = evaluate(runs, REQUIRED)
    assert verdict["failing"] == ["migration-policy-check"]
    assert verdict["failing_required"] == []
    assert verdict["failing_not_required"] == ["migration-policy-check"]


def test_only_the_named_advisory_bots_are_excluded() -> None:
    """The exclusion is a short literal list, not a category.

    Asserted so nobody widens it to "bots" or "anything not required" — the
    second is exactly the rejected design.
    """
    assert ADVISORY_NAMES == {"Sourcery review", "CodeRabbit"}
    others = ["vulture", "deptry", "typos", "release-docs-check", "calibration"]
    runs = [*_all_required_green(), *[_run(n, "failure") for n in others]]
    assert evaluate(runs, REQUIRED)["failing"] == sorted(others)


def test_an_advisory_name_matching_nothing_is_reported() -> None:
    """A renamed bot must not silently start blocking again.

    There is no API that says which checks are advisory, so the list is
    literal and can rot. The run says so out loud instead.
    """
    runs = [*_all_required_green(), _run("Sourcery review", "failure")]
    assert evaluate(runs, REQUIRED)["advisory_unmatched"] == ["CodeRabbit"]


def test_red_required_blocks() -> None:
    runs = _all_required_green()
    runs = [r for r in runs if r["name"] != "pattern-scan"]
    runs.append(_run("pattern-scan", "failure"))
    assert evaluate(runs, REQUIRED)["failing"] == ["pattern-scan"]


@pytest.mark.parametrize("conclusion", ["failure", "timed_out", "action_required"])
def test_every_failing_conclusion_blocks(conclusion: str) -> None:
    """`cancelled` is deliberately absent from this list — see #632 below."""
    runs = [r for r in _all_required_green() if r["name"] != "secrets-scan"]
    runs.append(_run("secrets-scan", conclusion))
    assert evaluate(runs, REQUIRED)["failing"] == ["secrets-scan"]


# --- inherited behaviour that must survive ------------------------------


@pytest.mark.parametrize("stale_first", [False, True])
def test_superseded_cancelled_run_does_not_block(stale_first: bool) -> None:
    """#632: keep the newest row per name, and cancelled is not a failure.

    Both list orders are exercised, because either one alone is passed by a
    positional implementation: with the stale row last, first-wins survives;
    with it first, last-wins survives. Only sorting on `started_at` passes
    both.

    `pytest (3.12)` is a floor name, so a lone `cancelled` row for it now
    fails: `failing == []` holds only if the newer success is picked. The
    assertion on `latest_per_name` pins the row directly.
    """
    fresh = _run("pytest (3.12)", "success", started_at="2026-08-06T11:00:00Z")
    stale = _run("pytest (3.12)", "cancelled", started_at="2026-08-06T10:00:00Z")
    runs = [r for r in _all_required_green() if r["name"] != "pytest (3.12)"]
    runs.extend([stale, fresh] if stale_first else [fresh, stale])

    assert evaluate(runs, REQUIRED)["failing"] == []
    assert latest_per_name(runs)["pytest (3.12)"]["conclusion"] == "success"


def test_a_superseded_failure_is_also_dropped_by_the_dedup() -> None:
    """The direction where the dedup is load-bearing for `failing`.

    Here the stale row is a genuine `failure`, so taking the wrong row per
    name puts a green context into `failing` and unlabels a mergeable PR.
    """
    runs = [r for r in _all_required_green() if r["name"] != "history-scan"]
    runs.append(_run("history-scan", "failure", started_at="2026-08-06T10:00:00Z"))
    runs.append(_run("history-scan", "success", started_at="2026-08-06T11:00:00Z"))

    assert evaluate(runs, REQUIRED)["failing"] == []


@pytest.mark.parametrize("name", ["e2e (pipx)", "mutation (advisory, PR diff)"])
def test_a_lone_cancellation_outside_the_floor_is_not_a_failure(name: str) -> None:
    """Preserved from the original for every name outside the floor.

    The mutation diff job ends `cancelled` at its 60-minute cap on routine
    PRs, so counting that as failing would bounce a PR on an advisory job.
    """
    assert name not in FLOOR
    runs = [*_all_floor_green(), _run(name, "cancelled")]
    assert evaluate(runs, REQUIRED)["failing"] == []


@pytest.mark.parametrize("name", sorted(FLOOR))
def test_a_lone_cancellation_of_a_floor_name_blocks(name: str) -> None:
    """A job that hits `timeout-minutes` reports `cancelled`, not `timed_out`.

    With the `pytest (*)` legs capped at 30 minutes, a hung leg is the only
    row for its name and is `cancelled`. Not counting it left `failing`,
    `pending` and `missing` all empty, and the train pushed.
    """
    runs = [r for r in _all_floor_green() if r["name"] != name]
    runs.append(_run(name, "cancelled"))
    verdict = evaluate(runs, REQUIRED)
    assert verdict["failing"] == [name]
    assert verdict["pending"] == [] and verdict["missing"] == []


def test_a_cancelled_required_context_outside_floor_names_blocks() -> None:
    """The floor is `required | FLOOR_NAMES`, so a ruleset-only context counts."""
    extra = "required-only-context"
    assert extra not in FLOOR_NAMES
    runs = [*_all_floor_green(), _run(extra, "cancelled")]
    assert evaluate(runs, REQUIRED | {extra})["failing"] == [extra]


def test_both_timed_out_pytest_legs_fail_through_main(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """The reported case through the CLI: both required legs cancelled."""
    runs = [
        r for r in _all_floor_green()
        if r["name"] not in {"pytest (3.12)", "pytest (3.13)"}
    ]
    runs += [_run("pytest (3.12)", "cancelled"), _run("pytest (3.13)", "cancelled")]
    rollup = tmp_path / "rollup.json"
    rules = tmp_path / "rules.json"
    rollup.write_text(json.dumps({"check_runs": runs}))
    rules.write_text(json.dumps(_rules(sorted(REQUIRED))))
    assert main(["--rollup", str(rollup), "--rules", str(rules)]) == 0
    verdict = json.loads(capsys.readouterr().out)
    assert verdict["failing_required"] == ["pytest (3.12)", "pytest (3.13)"]


def test_the_trains_own_jobs_are_excluded() -> None:
    """Waiting on itself would deadlock."""
    runs = [
        *_all_required_green(),
        _run("Attempt merge-train FF", None, status="in_progress"),
        _run("merge", None, status="in_progress"),
    ]
    assert evaluate(runs, REQUIRED)["pending"] == []
    assert "merge" not in latest_per_name(runs)


# --- pending is scoped to the required set ------------------------------


def test_a_slow_advisory_bot_does_not_hold_the_train() -> None:
    runs = [*_all_required_green(), _run("Sourcery review", None, status="in_progress")]
    assert evaluate(runs, REQUIRED)["pending"] == []


def test_a_slow_non_advisory_check_does_hold_the_train() -> None:
    """The other half of the same property — it must not over-narrow."""
    runs = [*_all_required_green(), _run("e2e", None, status="queued")]
    assert evaluate(runs, REQUIRED)["pending"] == ["e2e"]


_MUTATION = "mutation (advisory, PR diff)"


@pytest.mark.parametrize("status", ["queued", "in_progress", "pending"])
def test_a_running_mutation_job_does_not_hold_the_train(status: str) -> None:
    """#1633: its 60-minute run outlasted the train's 10-minute wait."""
    runs = [*_all_required_green(), _run(_MUTATION, None, status=status)]
    verdict = evaluate(runs, REQUIRED)
    assert verdict["pending"] == []
    assert verdict["pending_not_gating"] == [_MUTATION]


def test_a_finished_or_absent_mutation_job_is_not_logged_as_running() -> None:
    finished = [*_all_required_green(), _run(_MUTATION, "success")]
    assert evaluate(finished, REQUIRED)["pending_not_gating"] == []
    assert evaluate(_all_required_green(), REQUIRED)["pending_not_gating"] == []


@pytest.mark.parametrize(
    "name", ["migration-policy-check", "mutation", "mutmut"],
)
def test_a_running_check_with_a_similar_name_still_holds_the_train(
    name: str,
) -> None:
    """The exemption is one exact name, not a pattern."""
    runs = [*_all_required_green(), _run(name, None, status="in_progress")]
    assert evaluate(runs, REQUIRED)["pending"] == [name]


def test_a_failed_mutation_job_still_blocks() -> None:
    """Only its running state is exempt. Job-level `continue-on-error` keeps
    the workflow run green but leaves the job's check-run at `failure`, so a
    failure read while the train polls blocks it."""
    runs = [*_all_required_green(), _run(_MUTATION, "failure")]
    assert evaluate(runs, REQUIRED)["failing"] == [_MUTATION]


_SIZE_CHECK = "size-check"


@pytest.mark.parametrize("status", ["queued", "in_progress", "pending"])
def test_a_running_size_check_does_not_hold_the_train(status: str) -> None:
    """#1786: `pr-size-soft-cap.yml` triggers on `labeled`, so adding
    `ready-to-merge` starts a fresh advisory run on the head at the moment
    the train begins polling, and the train waited on it every time."""
    runs = [r for r in _all_floor_green() if r["name"] != _SIZE_CHECK]
    runs.append(_run(_SIZE_CHECK, None, status=status))
    verdict = evaluate(runs, REQUIRED)
    assert verdict["pending"] == []
    assert verdict["missing"] == []
    assert verdict["pending_not_gating"] == [_SIZE_CHECK]


def test_a_failed_size_check_still_blocks() -> None:
    """Only its running state is exempt; the result still gates."""
    runs = [r for r in _all_floor_green() if r["name"] != _SIZE_CHECK]
    runs.append(_run(_SIZE_CHECK, "failure"))
    assert evaluate(runs, REQUIRED)["failing"] == [_SIZE_CHECK]


def test_a_size_check_that_never_reported_is_still_missing() -> None:
    """It stays in the floor: exempt from the wait once it has a row,
    missing until it does."""
    runs = [r for r in _all_floor_green() if r["name"] != _SIZE_CHECK]
    assert evaluate(runs, REQUIRED)["missing"] == [_SIZE_CHECK]


_TEST_OUTCOMES = "test-outcomes"


@pytest.mark.parametrize("status", ["queued", "in_progress", "pending"])
def test_a_running_test_outcomes_job_does_not_hold_the_train(status: str) -> None:
    """#1829: the job `needs: pytest`, so it runs after every other gate
    has settled, and waiting on it would add its whole run to every merge."""
    runs = [*_all_floor_green(), _run(_TEST_OUTCOMES, None, status=status)]
    verdict = evaluate(runs, REQUIRED)
    assert verdict["pending"] == []
    assert verdict["pending_not_gating"] == [_TEST_OUTCOMES]


def test_test_outcomes_is_outside_the_floor() -> None:
    """Advisory, so a head without its row is not `missing`, and a
    `cancelled` row from its job cap does not fail the head."""
    assert _TEST_OUTCOMES not in FLOOR_NAMES
    assert evaluate(_all_floor_green(), REQUIRED)["missing"] == []
    runs = [*_all_floor_green(), _run(_TEST_OUTCOMES, "cancelled")]
    assert evaluate(runs, REQUIRED)["failing"] == []


def test_the_nongating_set_is_exactly_the_advisory_jobs() -> None:
    """A literal list, so widening it is a visible edit here."""
    assert NONGATING_WHILE_PENDING == {_MUTATION, _SIZE_CHECK, _TEST_OUTCOMES}


def _job_bodies(workflow: Path) -> dict[str, list[str]]:
    """`{check-run name: its own key lines}` for every job.

    The check-run name is the job's `name:` key when it has one and its id
    otherwise, which is how GitHub names the row. A job is a two-space-
    indented key under `jobs:`; its own keys are the lines indented exactly
    four spaces, so a step's `name:` (indented further) never counts, and a
    commented line never matches.
    """
    jobs: dict[str, list[str]] = {}
    body: list[str] | None = None
    job_id = ""
    in_jobs = False

    def flush() -> None:
        if body is not None:
            names = [ln[len("    name: "):] for ln in body if ln.startswith("    name: ")]
            jobs[(names[0] if names else job_id).strip()] = body

    for line in workflow.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" "):
            flush()
            body = None
            in_jobs = line.rstrip() == "jobs:"
            continue
        if in_jobs and re.match(r"^  [A-Za-z0-9_-]+:\s*$", line):
            flush()
            body = []
            job_id = line.strip()[:-1]
            continue
        if body is not None and line.startswith("    ") and line[4] != " ":
            body.append(line.rstrip())
    flush()
    return jobs


def test_every_nongating_name_is_a_job() -> None:
    """A renamed job would silently be waited on again."""
    jobs: dict[str, list[str]] = {}
    for wf in sorted((_REPO / ".github" / "workflows").glob("*.yml")):
        jobs.update(_job_bodies(wf))
    for name in NONGATING_WHILE_PENDING:
        assert name in jobs, f"no job is named {name!r}"


def _prose(path: Path) -> str:
    """A file's prose, joined into one line.

    For a workflow, every `#` comment. For Python, every line with a leading
    `#` removed, so comments, docstrings and test names are all read.
    """
    lines = path.read_text().splitlines()
    if path.suffix == ".py":
        parts = [ln.strip().removeprefix("#").strip() for ln in lines]
    else:
        parts = [ln.split("#", 1)[1].strip() for ln in lines if "#" in ln]
    return " ".join(p for p in parts if p)


# Each wording this repo has used for the claim that job-level
# `continue-on-error` keeps the job's check-run off `failure`.
_KEEPS_A_CHECK_OFF_FAILURE = re.compile(
    r"continue-on-error`? keeps the (?:job's )?conclusion off"
    r"|keeps the job's failure off the train"
    r"|check-run conclusion is `?success`? even when a step fails"
    r"|`continue-on-error` is what keeps the two consistent"
    r"|cannot_block_the_merge_train"
)

_CONTINUE_ON_ERROR_PROSE = sorted(
    [
        "scripts/merge_train_gate.py",
        "tests/test_mutation_workflow_1457.py",
        *(
            p.relative_to(_REPO).as_posix()
            for p in (_REPO / ".github" / "workflows").glob("*.yml")
        ),
    ]
)


@pytest.mark.parametrize("path", _CONTINUE_ON_ERROR_PROSE)
def test_no_comment_says_continue_on_error_keeps_a_check_off_failure(path: str) -> None:
    """Job-level `continue-on-error` changes the workflow run's conclusion,
    not the job's check-run, which is the row the gate reads. A comment that
    says otherwise sends a reader to the wrong reason a nongating job's late
    failure is ignored."""
    claim = _KEEPS_A_CHECK_OFF_FAILURE.search(_prose(_REPO / path))  # path-filter: scripts/merge_train_gate.py tests/test_mutation_workflow_1457.py .github/workflows/**
    assert claim is None, f"{path}: {claim.group(0)!r}"


# One `||` expression behind an `&&`, over the file folded onto one line.
# Matched as two separate arms, an `if` that joined them with `&&` (never
# true) or hung them off its guard with `||` (true on every pull_request
# event) read as the same gate.
_LABEL_GATE = re.compile(
    r"&& \( "
    r"\(github\.event\.action == 'labeled' && github\.event\.label\.name == '([^']+)'\) "
    r"\|\| \(github\.event\.action == 'synchronize' && "
    r"contains\(github\.event\.pull_request\.labels\.\*\.name, '([^']+)'\)\) "
    r"\)"
)


def _label_gate(text: str) -> tuple[set[str], set[str]]:
    """The labels a workflow's `if` expressions fire on.

    Returns `(added by this event, present on a push)`. Comment lines are
    dropped first, so a comment that quotes the expression never counts.
    """
    code = " ".join(
        line.strip() for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    hits = _LABEL_GATE.findall(code)
    return {added for added, _ in hits}, {present for _, present in hits}


def _named_job_if(text: str, name: str) -> str:
    """The folded `if:` block of the job whose `name:` is *name*."""
    lines = text.splitlines()
    at = lines.index(f"    name: {name}")
    start = max(i for i in range(at) if re.match(r"^  \S", lines[i]))
    end = next(
        (i for i in range(at, len(lines)) if re.match(r"^  \S", lines[i])),
        len(lines),
    )
    return _job_if("\n".join(lines[start:end]))


def _mutation_source() -> Path:
    """The workflow that defines the mutation diff job.

    The checks that read it are about that job alone. `size-check` is also
    in NONGATING_WHILE_PENDING, but its `if` skips only drafts and its
    workflow has neither a `paths:` filter nor a `skip=true` exit, so
    neither check describes it.
    """
    return next(
        path for path in sorted(_WORKFLOW_DIR.glob("*.yml"))
        if _MUTATION in _job_bodies(path)
    )


def test_the_nongating_job_starts_on_the_trains_own_label() -> None:
    """#1786: the mutation diff job fires on `ready-to-merge`, as the train does.

    The two start on the same event, so a run that reaches mutmut can be
    pending at every poll the train makes, and NONGATING_WHILE_PENDING is
    what lets it run then without holding the train. The job's `if` is the
    train's whole condition, fork guard included, behind an event-name
    check, and the trigger types are read from both files. A rename or a
    dropped event in one of them fails here instead of leaving a job that
    never runs, and a dropped guard fails instead of leaving one that runs on
    PRs the train skips.
    """
    assert _MUTATION in NONGATING_WHILE_PENDING
    train_text = _WORKFLOW.read_text()
    assert _label_gate(train_text) == ({"ready-to-merge"}, {"ready-to-merge"})
    train_if = _job_if(train_text)
    train_types = {
        frozenset(types) for _, types in _pull_request_triggers(train_text)
    }
    assert train_types == {frozenset({"labeled", "synchronize"})}, train_types
    source = _mutation_source()
    text = source.read_text()
    job_if = _named_job_if(text, _MUTATION)
    assert job_if == f"github.event_name == 'pull_request' && {train_if}", (
        source.name, job_if,
    )
    types = {frozenset(types) for _, types in _pull_request_triggers(text)}
    assert types == train_types, f"{source.name} triggers on {types}"


def _nongating_comment() -> str:
    """The comment block right above NONGATING_WHILE_PENDING, folded."""
    lines = (_REPO / "scripts" / "merge_train_gate.py").read_text().splitlines()
    at = next(
        i for i, ln in enumerate(lines)
        if ln.startswith("NONGATING_WHILE_PENDING")
    )
    start = at
    while start > 0 and lines[start - 1].startswith("#"):
        start -= 1
    return " ".join(ln.lstrip("#").strip() for ln in lines[start:at])


# The scope step's early exit: no mutable change, so every mutmut step skips.
_EARLY_SKIP = re.compile(
    r'echo "skip=true" >> "\$\{GITHUB_OUTPUT\}"\n\s*exit 0\n'
)


def test_the_nongating_comment_names_the_runs_that_are_never_pending() -> None:
    """#1786: the job is not pending at every poll, and the comment says so.

    A diff that misses the job's `paths:` filter starts no run, and a scope
    step that finds no mutable change writes `skip=true` and finishes in
    minutes. The comment above NONGATING_WHILE_PENDING names each case its
    workflow has and claims none it lacks, so a dropped filter or early exit
    fails here until the comment changes with it.
    """
    comment = _nongating_comment()
    source = _mutation_source()
    text = source.read_text()
    has_paths = any(p for p, _ in _pull_request_triggers(text))
    assert ("`paths:`" in comment) == has_paths, (source.name, has_paths)
    early_skip = _EARLY_SKIP.search(text) is not None
    assert ("`skip=true`" in comment) == early_skip, (source.name, early_skip)


def test_a_later_skipped_row_does_not_hide_the_nongating_run() -> None:
    """#1786: a label added after ready-to-merge leaves a newer skipped row.

    The mutation diff job starts on every `labeled` event and its `if` skips
    it for other labels, so the SHA carries the real run and a later
    `skipped` row under one name. The real run's state still reaches the
    verdict, whether it is running or has failed.
    """
    late_skip = _run(_MUTATION, "skipped", started_at="2026-08-06T10:03:00Z")
    running = _run(_MUTATION, None, status="in_progress")
    verdict = evaluate([*_all_floor_green(), running, late_skip], REQUIRED)
    assert verdict["pending_not_gating"] == [_MUTATION]
    assert verdict["pending"] == []

    failed = _run(_MUTATION, "failure")
    verdict = evaluate([*_all_floor_green(), failed, late_skip], REQUIRED)
    assert verdict["failing"] == [_MUTATION]


def test_a_skipped_floor_row_still_replaces_a_cancelled_one() -> None:
    """The skipped-row rule is scoped to NONGATING_WHILE_PENDING names.

    A floor name whose newest row is `cancelled` gates, so widening the rule
    to every name would let an older cancelled row block a head whose later
    run skipped the job.
    """
    runs = [r for r in _all_floor_green() if r["name"] != "label"]
    runs.append(_run("label", "cancelled"))
    runs.append(_run("label", "skipped", started_at="2026-08-06T10:03:00Z"))
    assert evaluate(runs, REQUIRED)["failing"] == []


def test_a_slow_required_check_does_hold_the_train() -> None:
    runs = [r for r in _all_required_green() if r["name"] != "pytest (3.13)"]
    runs.append(_run("pytest (3.13)", None, status="in_progress"))
    assert evaluate(runs, REQUIRED)["pending"] == ["pytest (3.13)"]


def test_a_required_context_that_never_reported_is_missing_not_pending() -> None:
    """`missing` and `pending` are different and must not be merged.

    Pending resolves on its own; missing may never. Collapsing them would
    either hang the train to its timeout or, worse, let a required context
    that never ran pass as absent-and-therefore-fine.
    """
    runs = [r for r in _all_floor_green() if r["name"] != "secrets-scan"]
    verdict = evaluate(runs, REQUIRED)
    assert verdict["missing"] == ["secrets-scan"]
    assert verdict["pending"] == []


# --- resolution of the required set -------------------------------------


def test_required_contexts_are_read_from_the_branch_rules() -> None:
    assert required_contexts(_rules(sorted(REQUIRED))) == REQUIRED


def test_contexts_from_several_rulesets_are_unioned() -> None:
    """More than one ruleset can apply to a branch; first-match would lose one."""
    rules = [*_rules(["secrets-scan"]), *_rules(["pytest (3.12)"])]
    assert required_contexts(rules) == {"secrets-scan", "pytest (3.12)"}


def test_an_unresolvable_required_set_aborts(tmp_path: Path) -> None:
    """Losing the required set removes the presence floor, so it is fatal.

    This deliberately reverses the earlier behaviour, and the reason is
    #1435. While the required set only *labelled* failures, losing it cost
    nothing and aborting would have bricked merges on a ruleset edit. Now
    `missing` gates, and `missing` is derived from that set — so an empty set
    yields an empty `missing`, which silently removes the only signal here
    that is not an absence-test.

    The failure mode being prevented is specifically a quiet one: with no
    required set, a rollup of nothing at all produces a verdict whose every
    field is empty, which the workflow reads as green.
    """
    rollup = tmp_path / "rollup.json"
    rollup.write_text(json.dumps({"check_runs": []}))
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps([{"type": "pull_request", "parameters": {}}]))

    assert main(["--rollup", str(rollup), "--rules", str(rules)]) == 2


def test_an_empty_rollup_is_all_missing_not_all_green() -> None:
    """The #1435 defect, at the level the gate can see it.

    Every other field in the verdict is defined as an absence, so a head SHA
    with no check-runs produces `failing: []` and `pending: []` — indis-
    tinguishable from a fully green head. Only `missing` separates them.
    """
    verdict = evaluate([], REQUIRED)
    assert verdict["failing"] == []
    assert verdict["pending"] == []
    assert verdict["missing"] == sorted(FLOOR), (
        "an empty rollup must report every floor name as missing; "
        "without that the workflow cannot tell it from a green one"
    )


def test_advisory_bots_alone_do_not_satisfy_the_floor() -> None:
    """The exact shape eight PRs carried during the 2026-08-06 outage.

    A skipped Sourcery review and a green CodeRabbit are the entire rollup.
    Neither is gating, so `failing` and `pending` are empty and the pre-#1435
    loop broke straight to the FF push.
    """
    runs = [
        _run("Sourcery review", "skipped"),
        _run("CodeRabbit", "success"),
    ]
    verdict = evaluate(runs, REQUIRED)
    assert verdict["failing"] == []
    assert verdict["pending"] == []
    assert verdict["missing"] == sorted(FLOOR)


def test_an_unreadable_payload_still_aborts(tmp_path: Path) -> None:
    """Fail-closed survives where it still means something."""
    rollup = tmp_path / "rollup.json"
    rollup.write_text(json.dumps("not a rollup"))
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps(_rules(sorted(REQUIRED))))

    assert main(["--rollup", str(rollup), "--rules", str(rules)]) == 2


def test_a_resolvable_set_exits_zero_and_emits_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    rollup = tmp_path / "rollup.json"
    rollup.write_text(json.dumps({"check_runs": _all_required_green()}))
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps(_rules(sorted(REQUIRED))))

    assert main(["--rollup", str(rollup), "--rules", str(rules)]) == 0
    assert json.loads(capsys.readouterr().out)["failing"] == []


# --- the workflow actually uses it --------------------------------------


def test_the_workflow_calls_the_gate_and_no_longer_greps_every_check_run() -> None:
    """A perfect gate nothing calls is the failure mode being fixed.

    Both halves asserted: the workflow invokes this script, and the inline
    filter it replaces is gone. Deleting only one of those leaves the defect.
    """
    text = _WORKFLOW.read_text()
    assert "scripts/merge_train_gate.py" in text
    assert "rules/branches/" in text, (
        "the workflow must resolve the required set at run time rather than "
        "hard-coding it, or it drifts from the ruleset"
    )
    assert not re.search(
        r'select\(\.c == "failure" or \.c == "timed_out"', text
    ), "the inline jq failure filter is still present"


def test_the_failure_message_distinguishes_required_from_advisory() -> None:
    """The second half of #1397: the message misattributed the cause.

    The old message said `required check(s) failed` while listing whatever
    had gone red, so an advisory bot was reported as a required context.
    The word `required` is now dropped from the unconditional part of the
    message and applied only to the failures that really are required, with
    the rest labelled as gating-but-not-required.
    """
    text = _WORKFLOW.read_text()
    assert ".failing_required | join" in text
    assert ".failing_not_required | join" in text
    assert "Not required by the ruleset but still gating" in text
    assert "required check(s) failed" not in text, (
        "the unconditional message must not call every failure required — "
        "that misattribution is the half of #1397 this fixes"
    )


def test_the_workflow_gates_on_failing_not_on_failing_required() -> None:
    """The design rejected in review must not come back in one word.

    `failing` covers every non-advisory check; `failing_required` is the
    required set alone. Swapping which one the workflow branches on is the
    entire required-only design — a one-token edit that demotes 19 real
    gates, including `migration-policy-check`. Nothing else in the suite
    distinguishes them, because both fields are correct in the verdict.
    """
    text = _WORKFLOW.read_text()
    assert re.search(r"fails=\$\(echo \"\$\{verdict\}\" \| jq -r '\.failing \| join", text), (
        "the gating branch must read .failing"
    )
    assert not re.search(r"fails=\$\(echo \"\$\{verdict\}\" \| jq -r '\.failing_required", text), (
        "gating on .failing_required is the required-only design; it demotes "
        "every non-required check to advisory"
    )


def test_the_workflow_waits_for_missing_required_contexts_too() -> None:
    """`missing` gates, and only the workflow can enforce that (#1435).

    `evaluate()` reports `missing`; the wait loop is what acts on it. Every
    module-level test here would still pass with the workflow breaking on
    `pending == 0` alone — which is precisely the state that would have
    FF-pushed a head carrying nothing but a skipped advisory bot.

    Asserted as the conjunction rather than as "the string `missing` appears",
    because it appeared in the old code too: as a warning printed on the way
    past it.
    """
    text = _WORKFLOW.read_text()
    assert re.search(
        r'if \[ "\$\{pending\}" = "0" \] && \[ "\$\{missing\}" = "0" \]; then',
        text,
    ), (
        "the wait loop must break only when pending AND missing are both "
        "zero; breaking on pending alone lets an empty rollup merge"
    )
    assert "the push will be rejected by branch protection" not in text, (
        "that claim is unverified in both directions -- the ruleset carries a "
        "`pull_request` rule this job demonstrably pushes past, and every "
        "push it has made carried green required contexts, so none of them "
        "discriminates. The gate must not lean on it either way"
    )


def test_the_gate_imports_only_the_standard_library() -> None:
    """The workflow runs it as bare `python3`, with no project deps installed.

    Asserted structurally rather than by spawning an interpreter: the test
    suite's `sys.executable` IS the project venv, so a subprocess run there
    would import a third-party module happily and prove nothing.
    """
    import ast

    source = (_REPO / "scripts" / "merge_train_gate.py").read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])

    non_stdlib = sorted(imported - sys.stdlib_module_names - {"__future__"})
    assert non_stdlib == [], (
        f"merge_train_gate.py imports non-stdlib module(s) {non_stdlib}; the "
        f"merge-train runner has no project dependencies installed"
    )


@pytest.mark.timeout(30)
def test_the_gate_script_is_invocable_as_a_script() -> None:
    """A smoke test that the file parses and argparse is wired.

    This does NOT establish the stdlib-only property — `sys.executable` is
    the project venv, where every dependency is importable. That property is
    asserted structurally by
    `test_the_gate_imports_only_the_standard_library`.

    Carries its own budget (#1307): a test that spawns a subprocess on the
    suite's 5 s default reports contention as a hang rather than as
    slowness. 30 s is an interpreter start plus an argparse `--help`.
    """
    proc = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "merge_train_gate.py"), "--help"],
        capture_output=True, text=True, check=False, timeout=20,
    )
    assert proc.returncode == 0, proc.stderr


# --- #1424: the base ref is checked, and a stacked PR is refused ----------
#
# The train's FF check (`git merge-base --is-ancestor origin/main <head>`)
# asks only whether the head *contains* main, which a stacked PR satisfies
# trivially — its parent branch is itself FF on main. Both directions are
# asserted below, because either alone is satisfied by a broken gate: one that
# refuses everything passes the stacked case, and one that refuses nothing
# passes the main case.


def test_a_main_based_pr_is_not_refused() -> None:
    """The gate must not refuse the ordinary case."""
    assert base_refusal("main", "main") is None


def test_a_stacked_pr_is_refused_and_the_message_names_the_base() -> None:
    """Refusal has to say which branch, or the author cannot act on it."""
    msg = base_refusal("docs/issue-1389-false-claims", "main")
    assert msg is not None
    assert "docs/issue-1389-false-claims" in msg
    assert "main" in msg


def test_the_refusal_says_the_required_checks_never_ran() -> None:
    """The actionable half.

    A stacked PR runs no required checks at all — `ci.yml` and
    `staging-gate.yml` declare `on: pull_request: branches: [main]`, so a PR
    based on a feature branch never matches their trigger. Telling the author
    only "wrong base" leaves them to rediscover that their green-looking check
    list is missing every gate.
    """
    msg = base_refusal("feature/x", "main")
    assert msg is not None
    assert "no required checks" in msg
    # And the retarget-alone trap, which costs a second round otherwise.
    assert "pull_request.edited" in msg


def test_an_unresolved_base_is_refused_rather_than_waved_through() -> None:
    """Fail closed. An empty base is what a failed `gh pr view` yields."""
    assert base_refusal("", "main") is not None


def test_the_default_branch_is_not_hard_coded() -> None:
    """A repo that renames its default branch must not refuse every PR."""
    assert base_refusal("trunk", "trunk") is None
    assert base_refusal("main", "trunk") is not None


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    ("base", "expected_code"), [("main", 0), ("feature/x", 3)],
)
def test_base_mode_exit_codes(base: str, expected_code: int) -> None:
    """The workflow branches on the exit code, so pin it end-to-end.

    Carries its own budget for the same reason as the smoke test above
    (#1307): a subprocess spawn on the suite's 5 s default reports
    contention as a hang rather than as slowness.
    """
    proc = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "merge_train_gate.py"),
         "--base-ref", base, "--default-branch", "main"],
        capture_output=True, text=True, check=False, timeout=20,
    )
    assert proc.returncode == expected_code, proc.stderr


def test_the_workflow_runs_the_base_check_before_waiting_for_checks() -> None:
    """Placement is the point, not merely presence.

    A stacked PR has no required checks to wait for, so a base check placed
    inside the poll loop would stall for the whole check wait instead of
    refusing immediately. Assert it precedes the wait step in the file.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert "--base-ref" in text, "the workflow never runs the base check"
    base_at = text.index("--base-ref")
    wait_at = text.index("waiting for required checks")
    assert base_at < wait_at, "the base check must run before the check wait"
    # And its failure must return the PR to the queue with an explanation.
    assert "fail_and_unlabel \"${base_msg}\"" in text
    # AC4: the resolved base is printed on every run, not only on refusal.
    assert "resolved base:" in text
    # The base must be READ FROM THE PR. Passing a constant would satisfy
    # every assertion above while defeating the check entirely — which is
    # what a mutation to `BASE_REF=main` does, and it survived until this
    # line existed.
    assert "--json baseRefName" in text, (
        "the workflow must resolve the base from the PR, not assume it"
    )
    assert '--base-ref "${BASE_REF}"' in text, (
        "the resolved base must be what is passed to the gate"
    )


# --- the presence floor covers the gating set (#1458) --------------------
#
# #1435 built the floor from the required set, which made a head with *no*
# check-runs distinguishable from a green one. It left the *partial* case: five
# required contexts against ~26 check-runs, so 21 gates could be absent and the
# verdict was byte-identical. #1436's `workflow_dispatch` hatch turned that from
# unconstructible into two `gh workflow run` calls.


# Workflow file -> the FLOOR_NAMES it emits. Kept beside the parser below, which
# asserts this mapping and `.github/workflows/` agree in BOTH directions: a new
# unfiltered PR workflow fails the guard until it is added here, and adding it
# here without adding its names to FLOOR_NAMES fails too.
_FLOOR_SOURCES: dict[str, frozenset[str]] = {
    "bench-smoke.yml": frozenset({"bench-smoke"}),
    "ci.yml": frozenset({
        "derived-figures (3.12)",
        "derived-figures (3.13)",
        "posterior-writers",
        "pytest (3.12)",
        "pytest (3.13)",
    }),
    "label-docs.yml": frozenset({"label"}),
    "migration-policy-check.yml": frozenset({"migration-policy-check"}),
    "pr-metadata.yml": frozenset({"pr-body-issue-link", "pr-title-prefix"}),
    "pr-size-soft-cap.yml": frozenset({"size-check"}),
    "staging-gate.yml": frozenset({
        "commit-msg-prefix",
        "history-scan",
        "pattern-scan",
        "release-docs-check",
        "secrets-scan",
    }),
    "typos.yml": frozenset({"typos"}),
    "zizmor.yml": frozenset({"zizmor"}),
}

# Rows an unfiltered PR workflow emits on every head that stay out of
# FLOOR_NAMES on purpose, per workflow. `test-outcomes` (ci.yml, #1829) is
# advisory: it is in NONGATING_WHILE_PENDING, and a floored `cancelled`
# row would gate on it. The drift guard subtracts exactly these names, so
# any other new job still fails it until FLOOR_NAMES carries it.
_UNFLOORED_ADVISORY: dict[str, frozenset[str]] = {
    "ci.yml": frozenset({"test-outcomes"}),
}

_WORKFLOW_DIR = _REPO / ".github" / "workflows"


# Both spellings emit a check-run on a PR head. `label-docs.yml` uses
# `pull_request_target` so it can label fork PRs, and a guard that knew only
# `pull_request` silently dropped it out of the derived set — which is the
# drift this guard exists to catch, so it must know every spelling.
_PR_EVENTS = frozenset({"pull_request", "pull_request_target"})
_ON_KEY = re.compile(r"""(on|"on"|'on'):(.*)""")
_FLOW_PATHS = re.compile(r"[{,]\s*paths(-ignore)?\s*:")
_FLOW_TYPES = re.compile(r"[{,]\s*types\s*:\s*(\[[^\]]*\]|[^,}]*)")


def _uncommented(raw: str) -> str:
    """An unquoted YAML value without its trailing comment."""
    return re.sub(r"(?:^|\s)#.*$", "", raw).strip()


def _flow_list(raw: str, what: str) -> list[str]:
    """The items of a one-line flow sequence such as `[opened, 'closed']`."""
    assert raw.startswith("[") and raw.endswith("]"), f"{what} shape not supported: {raw!r}"
    return [_scalar(v) for v in raw[1:-1].split(",") if v.strip()]


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _trigger(value: str, sub: list[str]) -> tuple[bool, list[str]]:
    """`(has_path_filter, types)` for one PR trigger, from its inline flow
    mapping `value` or, when that is empty, its indented block `sub`."""
    if value in {"", "{}", "null", "~"}:
        has_paths = any(re.match(r"\s+paths(-ignore)?:", s) for s in sub)
        typed = [m.group(1) for m in (re.match(r"\s+types:(.*)$", s) for s in sub) if m]
        return has_paths, _flow_list(_uncommented(typed[-1]), "types") if typed else []
    assert value.startswith("{") and value.endswith("}"), (
        f"pull_request shape not supported: {value!r}"
    )
    typed = _FLOW_TYPES.search(value)
    return bool(_FLOW_PATHS.search(value)), _flow_list(typed.group(1), "types") if typed else []


def _pull_request_triggers(text: str) -> list[tuple[bool, list[str]]]:
    """`(has_path_filter, types)` per PR trigger in the workflow's `on:` key.

    Empty when the workflow has no PR trigger at all. Reads every way GitHub
    accepts a trigger: an event name or a list of them on the `on:` line, a
    quoted `on` key, a block list of event names, and a block mapping whose
    PR entry is empty, an inline flow mapping, or an indented block. A shape
    outside those fails here, because reading it as no trigger would drop
    the workflow out of the drift guard. Parsed textually on purpose: PyYAML
    is not a dependency of this repo and is not importable in CI, so a guard
    that imported it would not run where it matters.

    Only the trigger's own entry is inspected. A `paths:` under `push:`
    does not path-filter the PR trigger, and reading the whole file would
    conflate the two.
    """
    lines = text.splitlines()
    on = next(((i, m) for i, ln in enumerate(lines) if (m := _ON_KEY.fullmatch(ln))), None)
    assert on is not None, "no top-level `on:` key was read"
    at, on_line = on
    inline = _uncommented(on_line.group(2))
    if inline:
        events = _flow_list(inline, "on") if inline.startswith("[") else [_scalar(inline)]
        return [(False, []) for event in events if event in _PR_EVENTS]

    block: list[str] = []
    for ln in lines[at + 1:]:
        if ln[:1].strip() and not ln.startswith("#"):
            break
        if ln.strip() and not ln.lstrip().startswith("#"):
            block.append(ln)
    assert block, "the `on:` key holds nothing"
    indent = _indent(block[0])

    found: list[tuple[bool, list[str]]] = []
    for i, ln in enumerate(block):
        if _indent(ln) != indent:
            continue
        entry = ln.strip()
        if entry.startswith("- "):
            if _scalar(entry[2:]) in _PR_EVENTS:
                found.append((False, []))
            continue
        key, sep, value = entry.partition(":")
        assert sep, f"`on:` entry shape not supported: {entry!r}"
        if _scalar(key) not in _PR_EVENTS:
            continue
        sub_end = next(
            (j for j in range(i + 1, len(block)) if _indent(block[j]) <= indent),
            len(block),
        )
        found.append(_trigger(_uncommented(value), block[i + 1 : sub_end]))
    return found


def _emits_a_row_on_every_pr_head(text: str) -> bool:
    """Does this workflow put a check-run on every push to a PR branch?

    Three ways to be excluded, all of them real in this repo: a path filter
    (the workflow may not run on a given diff), a `types:` list without
    `synchronize` (`auto-add-to-board` is `[opened]`, `auto-status-board` is
    `[closed]`), or no PR trigger at all (`replay-soak` is `schedule:`-only).

    Any one qualifying trigger is enough — a workflow carrying both spellings
    emits on every head if either of them does.
    """
    return any(
        not has_paths and (not types or "synchronize" in types)
        for has_paths, types in _pull_request_triggers(text)
    )


# A trailing comment is legal YAML and this repo writes one on two-space keys
# (`pull_request_target: # zizmor: ignore[...]`). A key regex without it read
# such a job as no job at all, and the workflow dropped out of the drift guard.
_JOB_KEY = re.compile(r"^  ([A-Za-z0-9_-]+):\s*(#.*)?$")
_MATRIX_AXIS = re.compile(r"^        ([A-Za-z0-9_-]+):\s*\[(.*)\]\s*$")
_SINGLE_QUOTED = re.compile(r"'((?:[^']|'')*)'\s*(?:#.*)?")
_DOUBLE_QUOTED = re.compile(r'"([^"\\]*)"\s*(?:#.*)?')
# Plain scalars that YAML 1.1 or 1.2 reads as null, a boolean, an integer,
# or a float. GitHub names a row after the parsed value, so an unquoted
# `3.10` names it `(3.1)`, and returning the text would name a row no head
# carries.
_NON_STRING_PLAIN = re.compile(
    r"~|null|Null|NULL"
    r"|true|True|TRUE|false|False|FALSE"
    r"|y|Y|yes|Yes|YES|n|N|no|No|NO|on|On|ON|off|Off|OFF"
    r"|[-+]?(?:0b[01_]+|0o[0-7_]+|0x[0-9a-fA-F_]+|[0-9][0-9_]*(?::[0-5]?[0-9])*)"
    r"|[-+]?(?:[0-9][0-9_]*)?\.[0-9_]*(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+"
    r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)"
)


def _scalar(raw: str) -> str:
    """A one-line YAML string scalar as YAML reads it: quotes removed, `''`
    inside single quotes read as `'`, and a trailing comment dropped. A
    shape this does not read, such as an escape inside double quotes, a
    block scalar, an anchor, or no value at all, fails rather than come back
    with its syntax still in it. So does an unquoted value that YAML reads
    as a number, a boolean, or null, because its text is not its name."""
    raw = raw.strip()
    m = _SINGLE_QUOTED.fullmatch(raw)
    if m:
        return m.group(1).replace("''", "'")
    m = _DOUBLE_QUOTED.fullmatch(raw)
    if m:
        return m.group(1)
    assert raw and raw[0] not in "'\"[]{}|>&*!%@`#", f"scalar shape not supported: {raw!r}"
    value = re.split(r"\s#", raw, maxsplit=1)[0].rstrip()
    assert not _NON_STRING_PLAIN.fullmatch(value), (
        f"unquoted {value!r} is not a string in YAML; quote it"
    )
    return value


def _check_run_names(text: str) -> frozenset[str]:
    """The check-run names a workflow's jobs emit, read from its `jobs:` block.

    GitHub names a row after the job's `name:` value as YAML reads it, so
    without quotes or a trailing comment, or after its id when there is none. A job with a `strategy.matrix` of inline lists and no
    `name:` emits one row per combination, named `<id> (<v1>, <v2>, ...)`
    in axis order, which is how `pytest` becomes `pytest (3.12)`. Any other
    matrix shape, including an inline `strategy:` or `matrix:` value such as
    an expression, a `name:` holding an expression, a job that calls a
    reusable workflow through a job-level `uses:` (GitHub names its rows
    `<caller> / <callee job>`, which this does not read from the callee),
    or a two-space line that is not a job key fails here so the parser is
    extended rather than deriving a wrong name or skipping a job.
    """
    import itertools

    lines = text.splitlines()
    start = next(
        (i + 1 for i, ln in enumerate(lines) if ln.rstrip() == "jobs:"), None
    )
    if start is None:
        return frozenset()
    names: set[str] = set()
    job_id: str | None = None
    display: str | None = None
    axes: list[list[str]] = []
    in_matrix = False

    def flush() -> None:
        if job_id is None:
            return
        base = display if display is not None else job_id
        assert "${{" not in base, f"{job_id}: a `name:` expression is not expanded here"
        if not axes:
            names.add(base)
            return
        for combo in itertools.product(*axes):
            names.add(f"{base} ({', '.join(combo)})")

    for ln in lines[start:]:
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if ln[:1].strip():
            break
        m = _JOB_KEY.match(ln)
        if m:
            flush()
            job_id, display, axes, in_matrix = m.group(1), None, [], False
            continue
        assert ln.startswith("   "), f"job key shape not supported: {ln.strip()}"
        assert not re.match(r"    uses\s*:", ln), (
            f"{job_id}: a reusable workflow call is not supported: {ln.strip()}"
        )
        if ln.startswith("    name:"):
            display = _scalar(ln[len("    name:"):])
        elif ln.startswith(("    strategy:", "      matrix:")):
            key, _, value = ln.strip().partition(":")
            assert re.fullmatch(r"\s*(#.*)?", value), (
                f"{job_id}: {key} shape not supported: {ln.strip()}"
            )
            in_matrix = key == "matrix"
        elif in_matrix and ln.startswith("        "):
            axis = _MATRIX_AXIS.match(ln)
            assert axis is not None, f"{job_id}: matrix shape not supported: {ln.strip()}"
            axes.append([_scalar(v) for v in axis.group(2).split(",") if v.strip()])
        elif in_matrix and not ln.startswith("        "):
            in_matrix = False
    flush()
    return frozenset(names)


def _derived_floor_sources(
    workflow_dir: Path = _WORKFLOW_DIR,
) -> dict[str, frozenset[str]]:
    """`{workflow file: check-run names}` for every workflow that emits a row
    on every PR head. The train's own job (SELF_NAMES) is dropped, so
    `merge-train.yml` contributes nothing. A workflow that emits a row but
    yields no job name fails here instead of dropping out of the comparison."""
    sources: dict[str, frozenset[str]] = {}
    for path in sorted([*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml")]):
        text = path.read_text()
        if not _emits_a_row_on_every_pr_head(text):
            continue
        emitted = _check_run_names(text)
        assert emitted, f"{path.name} runs on every PR head but no job name was read"
        names = emitted - SELF_NAMES
        if names:
            sources[path.name] = names
    return sources


def test_the_name_parser_expands_a_matrix_and_prefers_the_name_key() -> None:
    """The three shapes the floor's workflows use, on a synthetic file."""
    text = "\n".join([
        "on:",
        "  pull_request:",
        "jobs:",
        "  plain:",
        "    runs-on: ubuntu-latest",
        "  titled:",
        "    name: Shown Name",
        "    runs-on: ubuntu-latest",
        "  legs:",
        "    strategy:",
        "      matrix:",
        "        python-version: ['3.12', '3.13']",
        "    runs-on: ubuntu-latest",
        "",
    ])
    assert _check_run_names(text) == {
        "plain", "Shown Name", "legs (3.12)", "legs (3.13)",
    }


def test_the_name_parser_reads_a_job_key_with_a_trailing_comment() -> None:
    """A commented key is still a job, and the job after it is its own."""
    text = "\n".join([
        "jobs:",
        "  lint: # zizmor: ignore[unpinned-uses]",
        "    runs-on: ubuntu-latest",
        "  newjob:  # always runs",
        "    runs-on: ubuntu-latest",
        "",
    ])
    assert _check_run_names(text) == {"lint", "newjob"}


def test_the_name_parser_refuses_a_job_key_it_cannot_read() -> None:
    text = "jobs:\n  lint: {runs-on: ubuntu-latest}\n"
    with pytest.raises(AssertionError, match="job key shape not supported"):
        _check_run_names(text)


@pytest.mark.parametrize(
    "uses",
    [
        "    uses: ./.github/workflows/reusable-lint.yml",
        "    uses: owner/repo/.github/workflows/lint.yml@v1  # pinned",
    ],
)
@pytest.mark.parametrize("titled", [False, True])
def test_the_name_parser_refuses_a_reusable_workflow_call(uses: str, titled: bool) -> None:
    """GitHub names a called workflow's rows `<caller> / <callee job>`. Read
    as a plain job, it gives `call-lint` or `Lint`, a row no head carries, and
    a floor entry with that name waits on it until the train times out."""
    lines = ["jobs:", "  call-lint:", *(["    name: Lint"] if titled else []), uses, ""]
    with pytest.raises(AssertionError, match="reusable workflow call is not supported"):
        _check_run_names("\n".join(lines))


def test_the_name_parser_reads_a_step_level_uses() -> None:
    """A step's `uses:` is an action, not a reusable workflow call."""
    text = "\n".join([
        "jobs:",
        "  lint:",
        "    runs-on: ubuntu-latest",
        "    steps:",
        "      - uses: actions/checkout@v4",
        "        with:",
        "          fetch-depth: 0",
        "      - name: run",
        "        uses: ./.github/actions/lint",
        "",
    ])
    assert _check_run_names(text) == {"lint"}


@pytest.mark.parametrize(
    ("line", "name"),
    [
        ('    name: "Extra Job"', "Extra Job"),
        ("    name: 'Extra Job'", "Extra Job"),
        ("    name: 'it''s green'", "it's green"),
        ("    name: Extra Job  # shown in the checks list", "Extra Job"),
        ('    name: "Extra Job"  # quoted', "Extra Job"),
    ],
)
def test_the_name_parser_reads_a_name_the_way_yaml_does(line: str, name: str) -> None:
    """GitHub names the row after the parsed scalar, so quotes and a trailing
    comment are not part of it. `replay-soak-gate.yml` already quotes one."""
    text = f"jobs:\n  job:\n{line}\n    runs-on: ubuntu-latest\n"
    assert _check_run_names(text) == {name}


def test_the_name_parser_unquotes_matrix_values() -> None:
    text = "\n".join([
        "jobs:",
        "  legs:",
        "    strategy:",
        "      matrix:",
        "        python-version: [\"3.12\", '3.13', v3.14]",
        "",
    ])
    assert _check_run_names(text) == {"legs (3.12)", "legs (3.13)", "legs (v3.14)"}


@pytest.mark.parametrize(
    "value", ["3.10", "3.14", "1.0", "010", "1e3", "0x1f", "true", "False", "yes", "null", "~"]
)
def test_the_name_parser_refuses_a_matrix_value_yaml_reads_as_no_string(value: str) -> None:
    """YAML reads an unquoted `3.10` as the float 3.1, and GitHub names the
    row `legs (3.1)`. Taken as text it gives `legs (3.10)`, a row no head
    carries, and a floor entry with that name holds the train until it
    times out."""
    text = "\n".join([
        "jobs:",
        "  legs:",
        "    strategy:",
        "      matrix:",
        f"        python-version: ['3.12', {value}]",
        "",
    ])
    with pytest.raises(AssertionError, match="is not a string in YAML"):
        _check_run_names(text)


@pytest.mark.parametrize(
    "strategy",
    [
        ["    strategy:", "      matrix: ${{ fromJSON(needs.plan.outputs.legs) }}"],
        ["    strategy:", '      matrix: {python-version: ["3.12", "3.13"]}'],
        ["    strategy: {matrix: {python-version: ['3.12', '3.13']}}"],
    ],
)
def test_the_name_parser_refuses_a_matrix_it_cannot_expand(strategy: list[str]) -> None:
    """Read as no matrix, these give the bare job id, a row GitHub never
    emits; it emits `legs (3.12)` and `legs (3.13)`."""
    text = "\n".join(["jobs:", "  legs:", *strategy, "    runs-on: ubuntu-latest", ""])
    with pytest.raises(AssertionError, match="(matrix|strategy) shape not supported"):
        _check_run_names(text)


def test_the_name_parser_reads_a_matrix_key_with_a_trailing_comment() -> None:
    text = "\n".join([
        "jobs:",
        "  legs:",
        "    strategy:  # one leg per supported Python",
        "      matrix:  # axis order names the row",
        "        python-version: ['3.12']",
        "",
    ])
    assert _check_run_names(text) == {"legs (3.12)"}


@pytest.mark.parametrize(
    "line",
    ['    name: "a\\tb"', "    name: >", "    name: &id lint", "    name:"],
)
def test_the_name_parser_refuses_a_name_it_cannot_read(line: str) -> None:
    text = f"jobs:\n  job:\n{line}\n    runs-on: ubuntu-latest\n"
    with pytest.raises(AssertionError, match="scalar shape not supported"):
        _check_run_names(text)


_UNFILTERED_PR_WORKFLOW = "on:\n  pull_request:\n    branches: [main]\n"


@pytest.mark.parametrize(
    "on",
    [
        "on: [pull_request]",
        "on: [push, 'pull_request']",
        "on: pull_request",
        "on: pull_request_target  # labels fork PRs",
        '"on":\n  pull_request:',
        "'on':\n  pull_request:",
        "on:  # every PR head\n  pull_request:",
        "on:\n  - push\n  - pull_request",
        "on:\n  pull_request: {branches: [main]}",
        "on:\n  pull_request: {types: [opened, synchronize]}",
        "on:\n    pull_request:\n      branches: [main]",
    ],
)
def test_every_spelling_of_an_unfiltered_pr_trigger_emits_a_row(on: str) -> None:
    """GitHub accepts each of these, so a new always-run workflow written
    any of these ways must reach the drift guard."""
    assert _emits_a_row_on_every_pr_head(on + "\n")


@pytest.mark.parametrize(
    "on",
    [
        "on: [push]",
        "on: push",
        "on:\n  - push\n  - workflow_dispatch",
        "on:\n  pull_request: {paths: ['docs/**']}",
        "on:\n  pull_request: {types: [opened]}",
        "on:\n    pull_request:\n      paths-ignore: ['**.md']",
    ],
)
def test_a_filtered_or_absent_pr_trigger_emits_no_row(on: str) -> None:
    assert not _emits_a_row_on_every_pr_head(on + "\n")


@pytest.mark.parametrize(
    "text",
    [
        "on: {pull_request: {}}\n",
        "on:\n  pull_request:\n    types:\n      - opened\n",
        "on:\n  pull_request: [main]\n",
        "jobs: {}\n",
    ],
)
def test_the_trigger_parser_refuses_a_shape_it_cannot_read(text: str) -> None:
    """Fail rather than read the workflow as having no PR trigger, which
    would drop it out of the drift guard."""
    with pytest.raises(AssertionError, match="shape not supported|no top-level"):
        _pull_request_triggers(text)


def test_a_new_workflow_saved_as_yaml_is_derived(tmp_path: Path) -> None:
    (tmp_path / "lint.yaml").write_text(
        "on: [pull_request]\njobs:\n  lint:\n    runs-on: ubuntu-latest\n"
    )
    assert _derived_floor_sources(tmp_path) == {"lint.yaml": frozenset({"lint"})}


def test_a_new_workflow_with_a_commented_job_key_is_derived(tmp_path: Path) -> None:
    """The drift guard sees a new unfiltered workflow whose job key carries
    a comment, so leaving it out of FLOOR_NAMES fails the guard."""
    (tmp_path / "lint.yml").write_text(
        _UNFILTERED_PR_WORKFLOW
        + "jobs:\n  lint: # zizmor: ignore[unpinned-uses]\n    runs-on: ubuntu-latest\n"
    )
    assert _derived_floor_sources(tmp_path) == {"lint.yml": frozenset({"lint"})}


def test_a_workflow_whose_jobs_cannot_be_read_fails_the_derivation(
    tmp_path: Path,
) -> None:
    (tmp_path / "lint.yml").write_text(
        _UNFILTERED_PR_WORKFLOW + "jobs: {lint: {runs-on: ubuntu-latest}}\n"
    )
    with pytest.raises(AssertionError, match="no job name was read"):
        _derived_floor_sources(tmp_path)


def test_a_rollup_of_only_the_required_contexts_is_not_mergeable() -> None:
    """AC3 — the #1458 defect itself, at the level the gate can see it.

    `gh workflow run ci.yml` plus `gh workflow run staging-gate.yml` produces
    exactly the five required contexts. Before the floor widened, that head
    evaluated identically to a fully covered one: nothing failing, nothing
    pending, nothing missing.
    """
    verdict = evaluate(_all_required_green(), REQUIRED)
    assert verdict["failing"] == []
    assert verdict["pending"] == []
    assert verdict["missing"] != [], (
        "a head carrying only the five required contexts must not be mergeable; "
        "migration-policy-check, typos, bench-smoke and the rest never ran"
    )
    # Named because the module docstring names it: migration-policy-check is the
    # reason the gate covers every check-run, and it has no workflow_dispatch
    # trigger, so during the incident the hatch exists for it cannot be produced.
    assert "migration-policy-check" in verdict["missing"]


def test_a_docs_only_pr_still_passes_the_floor() -> None:
    """AC2 — the constraint that makes AC1 hard, demonstrated not argued.

    A docs-only diff never runs `e2e`, `CodeQL`, `deptry`, `vulture`,
    `Windows smoke` or `calibration` — their workflows are
    path-filtered, so they emit no check-run at all rather than a skipped one.
    A floor containing any of those names would block every docs PR forever.
    """
    runs = [*_all_floor_green(), _run("Sourcery review", "skipped")]
    verdict = evaluate(runs, REQUIRED)
    assert verdict["missing"] == []
    assert verdict["failing"] == []
    assert verdict["pending"] == []


@pytest.mark.parametrize("name", sorted(_FLOOR_SOURCES))
def test_no_floor_name_comes_from_a_path_filtered_workflow(name: str) -> None:
    """The property AC2 rests on, asserted per source rather than in aggregate.

    If any floor source acquires a `paths:` filter, docs-only PRs start hanging
    on a context that will never report — and the symptom would be a stuck
    train, not a test failure, without this.
    """
    text = (_WORKFLOW_DIR / name).read_text()
    assert _emits_a_row_on_every_pr_head(text), (
        f"{name} contributes to FLOOR_NAMES but no longer emits a check-run on "
        "every PR head; flooring on it will hang PRs it does not match"
    )


def test_every_unfiltered_pr_workflow_is_in_the_floor() -> None:
    """The drift guard AC1 requires.

    An explicit list is the only mechanism that names *which* check is absent,
    and the cost of an explicit list is that it goes stale silently. This
    re-derives the check-run names per workflow, so adding an unfiltered
    `pull_request` workflow fails here until its names are in FLOOR_NAMES.

    Compared by job name, not by workflow file (#1786). The file-level
    comparison this replaces passed while `ci.yml` had gained a third
    always-run job, `posterior-writers`, that the floor did not carry: the
    file was already in the mapping, so nothing noticed the new row.
    """
    derived = {
        name: rest
        for name, emitted in _derived_floor_sources().items()
        if (rest := emitted - _UNFLOORED_ADVISORY.get(name, frozenset()))
    }
    files = set(derived) | set(_FLOOR_SOURCES)
    drift = {
        name: (
            sorted(derived.get(name, frozenset()) - _FLOOR_SOURCES.get(name, frozenset())),
            sorted(_FLOOR_SOURCES.get(name, frozenset()) - derived.get(name, frozenset())),
        )
        for name in sorted(files)
        if derived.get(name) != _FLOOR_SOURCES.get(name)
    }
    assert not drift, (
        "the floor and .github/workflows/ have drifted; per workflow "
        f"(only in the workflow's jobs, only in the floor): {drift}"
    )


def test_every_floor_exemption_is_an_advisory_row_its_workflow_emits() -> None:
    """An exemption whose job was renamed or dropped would sit unused, and
    one for a name that waits or floors would hide a real gate."""
    derived = _derived_floor_sources()
    for file, names in _UNFLOORED_ADVISORY.items():
        assert names <= derived.get(file, frozenset()), (file, sorted(names))
        assert names <= NONGATING_WHILE_PENDING, sorted(names)
        assert not names & FLOOR_NAMES, sorted(names)


def test_the_floor_is_exactly_what_its_sources_emit() -> None:
    """The other direction: the mapping and the constant cannot disagree."""
    assert frozenset().union(*_FLOOR_SOURCES.values()) == FLOOR_NAMES


_FLOOR_ENTRY = re.compile(r'^\s+"(?P<name>[^"]+)",\s+#\s*(?P<file>\S+\.yml)\s*$')


def _floor_comment_attributions() -> dict[str, str]:
    """`{name: workflow file}` from the trailing comment on each FLOOR_NAMES
    entry in the gate's source."""
    source = (_REPO / "scripts" / "merge_train_gate.py").read_text()
    block = source[source.index("FLOOR_NAMES: frozenset[str] = frozenset({"):]
    block = block[: block.index("})")]
    found = {
        m.group("name"): m.group("file")
        for m in (_FLOOR_ENTRY.match(ln) for ln in block.splitlines())
        if m
    }
    assert set(found) == FLOOR_NAMES, "every floor entry carries a source comment"
    return found


def test_every_floor_comment_names_the_workflow_that_emits_the_row() -> None:
    """The comments are what a reader checks a missing row against (#1786).

    `pr-title-prefix` and `pr-body-issue-link` were attributed to
    `staging-gate.yml` after they moved to `pr-metadata.yml` in #1436, so a
    recovery that dispatched staging-gate looked complete while the two rows
    a dispatch cannot produce were still absent.
    """
    expected = {
        name: file for file, names in _FLOOR_SOURCES.items() for name in names
    }
    assert _floor_comment_attributions() == expected


_TABLE_ROW = re.compile(r"^\s*\| `(?P<file>[^`]+\.ya?ml)` \| (?P<names>.+?) \|\s*$")


def _contributing_recovery_table() -> dict[str, set[str]]:
    """`{workflow file: check-run names}` from the CONTRIBUTING.md recovery
    table. A file listed on more than one row gets the union of its rows."""
    rows: dict[str, set[str]] = {}
    for line in (_REPO / "CONTRIBUTING.md").read_text().splitlines():
        m = _TABLE_ROW.match(line)
        if m:
            names = re.findall(r"`([^`]+)`", m.group("names"))
            rows.setdefault(m.group("file"), set()).update(names)
    return rows


def test_the_recovery_table_lists_every_floor_row() -> None:
    """CONTRIBUTING.md's table is what a reader compares a head's rows
    against after a dispatch-only recovery (#1786). A floor row the table
    leaves out is one the reader never looks for."""
    table = _contributing_recovery_table()
    absent = {
        file: sorted(names - table.get(file, set()))
        for file, names in _derived_floor_sources().items()
        if names - table.get(file, set())
    }
    assert not absent, f"floor rows missing from the CONTRIBUTING.md table: {absent}"


def test_contributing_describes_the_floor_as_required_plus_floor_names() -> None:
    """The floor is `required | FLOOR_NAMES` (#1458), not the required set."""
    text = " ".join((_REPO / "CONTRIBUTING.md").read_text().split())
    assert "covers the required set only" not in text
    assert "is the required set plus `merge_train_gate.FLOOR_NAMES`" in text


def test_the_floor_excludes_advisory_bots_and_the_trains_own_jobs() -> None:
    """Flooring on either is a deadlock, not a gate.

    An advisory name in the floor would restore exactly the #1397 blocking the
    exclusion list exists to remove, and a self name would make the train wait
    on itself.
    """
    assert not (FLOOR_NAMES & ADVISORY_NAMES)
    assert not (FLOOR_NAMES & SELF_NAMES)


# --- #1786: each derived-figures matrix leg is a floor name -----------------
#
# `derived-figures` in `ci.yml` is a `python-version` matrix, so each leg
# reports its own check-run and each leg is its own floor row.

# The rows the matrix produces: job id `derived-figures`, no `name:`, one axis
# `python-version: ['3.12', '3.13']`. Written out rather than read from
# FLOOR_NAMES, so a floor that drops a leg fails here.
_DERIVED_LEGS = ("derived-figures (3.12)", "derived-figures (3.13)")


def _floor_green_with_derived_legs(
    legs: tuple[str, ...] = _DERIVED_LEGS,
) -> list[dict[str, Any]]:
    """A green floor that reports only the given matrix legs."""
    return [
        *(_run(name) for name in sorted(FLOOR - set(_DERIVED_LEGS))),
        *(_run(leg) for leg in legs),
    ]


@pytest.mark.parametrize("leg", _DERIVED_LEGS)
def test_one_derived_figures_leg_alone_leaves_the_other_leg_missing(leg: str) -> None:
    other = [n for n in _DERIVED_LEGS if n != leg]
    verdict = evaluate(_floor_green_with_derived_legs((leg,)), REQUIRED)
    assert verdict["missing"] == other


def test_neither_derived_figures_leg_leaves_both_legs_missing() -> None:
    verdict = evaluate(_floor_green_with_derived_legs(()), REQUIRED)
    assert verdict["missing"] == list(_DERIVED_LEGS)


@pytest.mark.parametrize("status", ["queued", "in_progress", "pending"])
@pytest.mark.parametrize("leg", _DERIVED_LEGS)
def test_a_running_derived_figures_leg_is_pending(leg: str, status: str) -> None:
    runs = [
        r for r in _floor_green_with_derived_legs() if r["name"] != leg
    ] + [_run(leg, None, status=status)]
    verdict = evaluate(runs, REQUIRED)
    assert verdict["pending"] == [leg]
    assert verdict["missing"] == []
    assert verdict["failing"] == []


@pytest.mark.parametrize("conclusion", ["failure", "cancelled"])
@pytest.mark.parametrize("leg", _DERIVED_LEGS)
def test_a_failed_or_cancelled_derived_figures_leg_fails(
    leg: str, conclusion: str,
) -> None:
    runs = [
        r for r in _floor_green_with_derived_legs() if r["name"] != leg
    ] + [_run(leg, conclusion)]
    verdict = evaluate(runs, REQUIRED)
    assert verdict["failing"] == [leg]
    assert verdict["missing"] == []


# --- #1695: the train runs only code from `main` -----------------------------


def _on_block(text: str) -> list[str]:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.rstrip() == "on:") + 1
    end = next(
        i for i in range(start, len(lines))
        if lines[i][:1].strip() and not lines[i].startswith("#")
    )
    return [ln for ln in lines[start:end] if ln.strip() and not ln.lstrip().startswith("#")]


def _job_if(text: str) -> str:
    """The `if:` block of the merge job, comments stripped."""
    lines = text.splitlines()
    i = next(n for n, ln in enumerate(lines) if ln.startswith("    if: |"))
    body: list[str] = []
    for ln in lines[i + 1:]:
        if not ln.startswith("      "):
            break
        body.append(ln.split("#")[0].strip())
    return " ".join(b for b in body if b)


def test_the_train_triggers_on_pull_request_target_only() -> None:
    """Under `pull_request` the workflow and its scripts came from the PR
    itself. Under `pull_request_target` they come from the default branch.
    No `branches:` filter, so a stacked PR still gets the #1424 refusal."""
    block = _on_block(_WORKFLOW.read_text())
    keys = [ln.split("#")[0].strip() for ln in block if re.match(r"^  \S", ln)]
    assert keys == ["pull_request_target:"], block
    assert not any(ln.strip().startswith("branches:") for ln in block), block


def test_the_fork_guard_gates_every_event() -> None:
    """`pull_request_target` gives a fork PR a write token. The guard must
    be joined to the event conditions with `&&`, in the `if:` itself."""
    cond = _job_if(_WORKFLOW.read_text())
    guard = "github.event.pull_request.head.repo.full_name == github.repository"
    assert cond.startswith(f"{guard} && ("), cond
    assert cond.endswith(")"), cond


def test_the_train_never_checks_out_the_pr_head() -> None:
    text = _WORKFLOW.read_text()
    assert text.count("uses: actions/checkout@") == 1
    checkout = text[text.index("uses: actions/checkout@"):]
    checkout = checkout[: checkout.index("\n      - ")]
    assert re.search(r"^\s+ref: main\s*$", checkout, re.M), checkout
    run = text[text.index("        run: |"):]
    code = "\n".join(ln.split("#")[0] for ln in run.splitlines())
    # The head is only ever named through env vars, never interpolated
    # into a script (template injection) ...
    assert "${{ github.event.pull_request.head" not in code
    # ... and never checked out, switched to, or reset onto.
    assert not re.search(
        r"git\s+(checkout|switch|reset|worktree\s+add)\b[^\n]*\$\{?HEAD_(SHA|REF)",
        code,
    )
    assert "refs/pull/" not in text


def test_concurrency_is_job_level() -> None:
    """A run whose `if` skips the job must not join the group, or it can
    displace a labelled run waiting for its slot."""
    text = _WORKFLOW.read_text()
    assert not re.search(r"^concurrency:", text, re.M)
    assert re.search(r"^    concurrency:\n      group: merge-train\n", text, re.M)


def test_the_head_ref_is_fetched_by_explicit_refspec() -> None:
    text = _WORKFLOW.read_text()
    assert '"+refs/heads/${HEAD_REF}:refs/remotes/origin/${HEAD_REF}"' in text
    assert 'git fetch origin main "${HEAD_REF}"' not in text


# --- #1786: the check wait fits inside the job, and every block is visible ---


def _check_timeout_seconds() -> int:
    match = re.search(r"^\s+CHECK_TIMEOUT_SECONDS: '(\d+)'\s*$", _WORKFLOW.read_text(), re.M)
    assert match is not None, "merge-train.yml no longer sets CHECK_TIMEOUT_SECONDS"
    return int(match.group(1))


def _job_timeout_minutes() -> int:
    match = re.search(r"^    timeout-minutes:\s*(\d+)\s*$", _WORKFLOW.read_text(), re.M)
    assert match is not None, "merge-train.yml no longer bounds the job"
    return int(match.group(1))


def test_the_job_timeout_outlasts_the_check_wait() -> None:
    """CI-14. The wait is the longest step. A job cap at or under it lets the
    runner kill the job mid-poll, with no comment and no unlabel, so the PR
    sits labeled with no feedback. The values are the 2026-10-21 ruling."""
    wait = _check_timeout_seconds()
    assert wait == 1200
    assert _job_timeout_minutes() == 25
    assert _job_timeout_minutes() * 60 > wait


def test_the_workflow_comments_quote_the_current_wait() -> None:
    """A comment that names a stale number sends a reader to the wrong
    deadline. Every `N-minute` wait or timeout the file mentions is the
    check wait."""
    minutes = _check_timeout_seconds() // 60
    mentions = re.findall(r"(\d+)-minute (?:wait|timeout)", _WORKFLOW.read_text())
    assert mentions, "the workflow no longer explains the wait in minutes"
    assert {int(m) for m in mentions} == {minutes}


# The slowest green pytest leg, rounded up: 15.6 minutes from run creation
# over 76 legs of PR runs of ci.yml from 2026-10-21 to 2026-11-02 (`gh run
# list --workflow ci.yml --event pull_request --status success --limit 40`,
# then each run's jobs, dropping legs under a minute).
_SLOWEST_GREEN_LEG_MINUTES = 16


def test_the_check_wait_comment_quotes_the_slowest_measured_leg() -> None:
    """The comment above the wait is where a reader judges its margin. One
    that quoted 11 to 14 minutes understated how close the slowest leg comes
    to it, and a wait cut to fit that range would bounce green PRs again."""
    text = _WORKFLOW.read_text()
    wait = re.search(r"^\s+CHECK_TIMEOUT_SECONDS: '", text, re.M)
    assert wait is not None
    comment: list[str] = []
    for line in reversed(text[: wait.start()].splitlines()):
        if not line.lstrip().startswith("#"):
            break
        comment.insert(0, line.strip().lstrip("#").strip())
    quoted = re.search(r"up to (\d+) minutes", " ".join(comment))
    assert quoted is not None, "the wait comment no longer quotes the slowest leg"
    assert int(quoted.group(1)) >= _SLOWEST_GREEN_LEG_MINUTES
    assert int(quoted.group(1)) * 60 < _check_timeout_seconds()


def _fail_and_unlabel_body() -> str:
    text = _WORKFLOW.read_text()
    start = text.index("fail_and_unlabel() {")
    return text[start: text.index("\n          }\n", start)]


_HEREDOC = re.compile(r"(?<!<)<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")
_EXIT = re.compile(r"(?<![\w$-])exit(?![\w-])")


def _shell_code(script: str) -> str:
    """`script` with its comments and heredoc bodies removed.

    A `#` starts a comment only outside quotes and at the start of a word,
    so `echo "PR #1"; exit 1` keeps its `exit`. Quote state carries across
    lines, because a single-quoted GraphQL query spans several. Heredoc
    bodies are prose, and an apostrophe in one would derail the quote state.
    """
    out: list[str] = []
    lines = script.split("\n")
    quote: str | None = None
    n = 0
    while n < len(lines):
        line = lines[n]
        n += 1
        kept: list[str] = []
        j = 0
        while j < len(line):
            c = line[j]
            if quote == "'":
                quote = None if c == "'" else quote
            elif c == "\\":
                kept.append(line[j: j + 2])
                j += 2
                continue
            elif quote == '"':
                quote = None if c == '"' else quote
            elif c in "'\"":
                quote = c
            elif c == "#" and (j == 0 or line[j - 1] in " \t;|&()"):
                break
            kept.append(c)
            j += 1
        code = "".join(kept)
        out.append(code)
        if quote is None:
            for m in _HEREDOC.finditer(code):
                while n < len(lines) and lines[n].strip() != m.group(2):
                    n += 1
                n += 1
    return "\n".join(out)


def _merge_step_script() -> str:
    """The merge step's `run: |` block, de-indented."""
    lines = _WORKFLOW.read_text().splitlines()
    start = lines.index("        run: |") + 1
    body: list[str] = []
    for ln in lines[start:]:
        if ln.strip() and not ln.startswith("          "):
            break
        body.append(ln[10:])
    return "\n".join(body)


@pytest.mark.parametrize(
    ("script", "count"),
    [
        ("exit 1", 1),
        ("[ -n \"$x\" ] && exit", 1),
        ('rc=1; exit "$rc"', 1),
        ('echo "[6/6] FF push #"; exit 1', 1),
        ("echo 'it is # here'; exit 2", 1),
        ('echo "a\\" # b"; exit 1', 1),
        ("# exit 1\necho done  # then exit", 0),
        ("q='a\nb'  # exit 1", 0),
        ("q='a\n#b'; exit 1", 1),
        ("cat <<EOF\nexit 1 is prose, it's here\nEOF\nexit 0", 1),
        ("echo $# ${#x}; exit 3", 1),
        ("echo exits; my-exit; exit_code=1", 0),
    ],
)
def test_the_exit_scanner_finds_every_exit_statement(script: str, count: int) -> None:
    assert len(_EXIT.findall(_shell_code(script))) == count


def test_every_block_path_emits_a_warning_annotation() -> None:
    """CI-05. A blocked train used to leave only a PR comment and a green
    run, so the Actions summary said nothing had happened. Every block path
    goes through `fail_and_unlabel`, which annotates the run as its first
    command, so no `gh` or `jq` call failing under `set -e` can end the step
    before the annotation. The step's only `exit` statement, in any form, is
    the one inside that function, so no path leaves the step around it."""
    statements = [
        ln.strip() for ln in _shell_code(_fail_and_unlabel_body()).splitlines() if ln.strip()
    ]
    commands = [s for s in statements[1:] if not s.startswith("local ")]
    assert re.match(r'echo "::warning[^\n]*\$\{reason\}', commands[0]), (
        "the ::warning annotation is not fail_and_unlabel's first command: "
        f"{commands[:2]}"
    )

    code = _shell_code(_merge_step_script())
    start = code.index("fail_and_unlabel() {")
    end = code.index("\n}\n", start)
    exits = [m.start() for m in _EXIT.finditer(code)]
    assert len(exits) == 1, (
        "a block path can bypass fail_and_unlabel: "
        f"{[code[i - 40: i + 12] for i in exits]}"
    )
    assert start < exits[0] < end, "the step's only exit is outside fail_and_unlabel"
