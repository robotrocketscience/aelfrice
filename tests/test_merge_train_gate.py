"""The merge-train gate reads the required set, not every check-run (#1397).

The defect this pins: `.github/workflows/merge-train.yml` enumerated every
check-run on the head SHA and unlabelled on any failure, so an advisory bot
could block a merge while the message said `required check(s) failed`. PR #1394
sat blocked behind a `Sourcery review` finding that was verified false.

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
    evaluate,
    latest_per_name,
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


def test_red_advisory_with_required_green_does_not_block() -> None:
    """The #1394 case. This is the whole point of the change."""
    runs = [*_all_required_green(), _run("Sourcery review", "failure")]
    verdict = evaluate(runs, REQUIRED)
    assert verdict["failing"] == []
    assert verdict["pending"] == []
    assert verdict["missing"] == []
    # Still reported, so it is visible rather than silently dropped.
    assert verdict["advisory_failing"] == ["Sourcery review"]


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


def test_superseded_cancelled_run_does_not_block(monkeypatch=None) -> None:
    """#632: keep the newest row per name, and cancelled is not a failure.

    The stale row is `cancelled` and *newer in list order* than the success,
    so a implementation that takes the first or last element rather than
    sorting on `started_at` fails here.
    """
    runs = [r for r in _all_required_green() if r["name"] != "pytest (3.12)"]
    runs.append(_run("pytest (3.12)", "success", started_at="2026-08-06T11:00:00Z"))
    runs.append(_run("pytest (3.12)", "cancelled", started_at="2026-08-06T10:00:00Z"))
    verdict = evaluate(runs, REQUIRED)
    assert verdict["failing"] == []
    assert latest_per_name(runs)["pytest (3.12)"]["conclusion"] == "success"


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


def test_an_empty_required_set_aborts_rather_than_merging(tmp_path: Path) -> None:
    """Fail-closed. The single most important assertion in this module.

    An unresolvable required set is indistinguishable from "the ruleset
    moved" or "the token lost read access". Reading it as "nothing is
    required" would merge anything — strictly worse than the over-blocking
    this change replaces.
    """
    rollup = tmp_path / "rollup.json"
    rollup.write_text(json.dumps({"check_runs": _all_required_green()}))
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps([{"type": "pull_request", "parameters": {}}]))

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
    ), "the inline all-check-runs failure filter is still present"


def test_the_failure_message_distinguishes_required_from_advisory() -> None:
    """The second half of #1397: the message misattributed the cause.

    `required check(s) failed` is kept — it is now *true*, because `fails`
    is drawn from the required set alone. What makes it honest is that a
    failing advisory check is reported alongside and explicitly labelled as
    not gating, rather than being folded into the same list under the same
    word.
    """
    text = _WORKFLOW.read_text()
    assert ".failing | join" in text, (
        "the failure list must come from the gate's required-only `failing` "
        "field, not from every check-run"
    )
    assert "advisory checks also failing, not gating" in text
    assert ".advisory_failing" in text


@pytest.mark.timeout(30)
def test_the_gate_script_runs_under_the_repo_python() -> None:
    """It runs on a runner with no dependencies installed beyond stdlib.

    Carries its own budget (#1307): a test that spawns a subprocess on the
    suite's 5 s default reports contention as a hang rather than as
    slowness. 30 s is an interpreter start plus an argparse `--help`.
    """
    proc = subprocess.run(
        [sys.executable, str(_REPO / "scripts" / "merge_train_gate.py"), "--help"],
        capture_output=True, text=True, check=False, timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
