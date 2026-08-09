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
    runs = [*_all_required_green(), _run("Sourcery review", "failure")]
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

    `failing == []` cannot carry this on its own — `cancelled` is not a
    failing conclusion, so that assertion is true whichever row is picked.
    The discriminating assertion is on `latest_per_name`.
    """
    fresh = _run("pytest (3.12)", "success", started_at="2026-08-06T11:00:00Z")
    stale = _run("pytest (3.12)", "cancelled", started_at="2026-08-06T10:00:00Z")
    runs = [r for r in _all_required_green() if r["name"] != "pytest (3.12)"]
    runs.extend([stale, fresh] if stale_first else [fresh, stale])

    assert evaluate(runs, REQUIRED)["failing"] == []
    assert latest_per_name(runs)["pytest (3.12)"]["conclusion"] == "success"


def test_a_superseded_failure_is_also_dropped_by_the_dedup() -> None:
    """The direction where the dedup is load-bearing for `failing`.

    Above, `failing == []` holds either way because `cancelled` is benign.
    Here the stale row is a genuine `failure`, so taking the wrong row per
    name puts a green context into `failing` and unlabels a mergeable PR.
    """
    runs = [r for r in _all_required_green() if r["name"] != "history-scan"]
    runs.append(_run("history-scan", "failure", started_at="2026-08-06T10:00:00Z"))
    runs.append(_run("history-scan", "success", started_at="2026-08-06T11:00:00Z"))

    assert evaluate(runs, REQUIRED)["failing"] == []


def test_a_genuine_lone_cancellation_is_not_a_failure_either() -> None:
    """Preserved from the original: cancelled surfaces, it does not block.

    Asserted so nobody "tightens" the gate by adding cancelled to the failing
    set — that would reintroduce the #632 false positive.
    """
    runs = [r for r in _all_required_green() if r["name"] != "history-scan"]
    runs.append(_run("history-scan", "cancelled"))
    assert evaluate(runs, REQUIRED)["failing"] == []


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
    runs = [r for r in _all_required_green() if r["name"] != "secrets-scan"]
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
    assert verdict["missing"] == sorted(REQUIRED), (
        "an empty rollup must report every required context as missing; "
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
    assert verdict["missing"] == sorted(REQUIRED)


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
    inside the poll loop would stall to the 30-minute timeout instead of
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
