"""Decide whether the merge-train may proceed, from check-runs (#1397).

The gate used to be inline `jq` in `.github/workflows/merge-train.yml`, and it
enumerated **every** check-run on the head SHA. That made `Sourcery review`,
CodeRabbit and every other advisory bot de facto merge-blocking, while the
failure message said `required check(s) failed` — naming a set the workflow
never read. PR #1394 sat blocked behind a verified-false SQL-injection finding
because of it.

This module gates on the branch's **actual** required contexts, resolved at run
time from `GET /repos/{owner}/{repo}/rules/branches/{branch}` so the workflow
cannot drift from the ruleset. Advisory results stay visible on the PR; they
stop deciding whether main moves.

It lives here rather than in the workflow because a gate that cannot be tested
is how the original defect survived. The decision is a pure function of
(rollup, required set) and is exercised directly by
`tests/test_merge_train_gate.py`, including the case that matters most — a red
advisory check with every required context green must **merge**.

**Fail-closed on an unresolvable required set.** An empty set would otherwise
mean "nothing is required", i.e. merge anything, which is strictly worse than
the over-blocking this replaces. Resolution returning nothing is treated as an
error, not as permission.

Two behaviours are inherited deliberately and must not be simplified away:

* **Per-name latest-run dedup (#632).** A superseded run leaves a `cancelled`
  row on the same SHA; keeping only the newest row per name is what stops it
  blocking merge. `cancelled` is therefore *not* a failure — after the dedup a
  genuine cancellation surfaces as the only row for its name and an operator
  can re-trigger.
* **`pending` is scoped to the required set too**, so a slow or silent advisory
  bot no longer holds the train to `CHECK_TIMEOUT_SECONDS`.

Usage::

    gh api "repos/${REPO}/commits/${SHA}/check-runs?per_page=100" > rollup.json
    gh api "repos/${REPO}/rules/branches/main" > rules.json
    python scripts/merge_train_gate.py --rollup rollup.json --rules rules.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# The train's own jobs. Waiting on them would deadlock: they are the
# thing doing the waiting.
SELF_NAMES: frozenset[str] = frozenset({"Attempt merge-train FF", "merge"})

FAILING_CONCLUSIONS: frozenset[str] = frozenset(
    {"failure", "timed_out", "action_required"}
)
PENDING_STATUSES: frozenset[str] = frozenset({"queued", "in_progress", "pending"})


def required_contexts(rules: list[dict[str, Any]]) -> set[str]:
    """The required status-check contexts a branch's rules declare.

    Reads the `rules/branches/{branch}` shape, which needs only read access —
    the `rulesets` endpoint needs admin, and the workflow runs with the default
    token. Multiple rulesets can apply to one branch, so contexts are unioned
    rather than taken from the first match.
    """
    contexts: set[str] = set()
    for rule in rules:
        if rule.get("type") != "required_status_checks":
            continue
        params = rule.get("parameters") or {}
        for check in params.get("required_status_checks") or []:
            context = check.get("context")
            if isinstance(context, str) and context:
                contexts.add(context)
    return contexts


def latest_per_name(check_runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Newest run per check name, excluding the train's own jobs (#632).

    GitHub leaves a row per attempt on the SHA, so a re-triggered check has a
    stale `cancelled` row alongside its real result. Sorting by `started_at`
    and keeping the last mirrors what GitHub's own required-checks evaluation
    does. Rows with no `started_at` sort first so a row that has one always
    wins over one that does not.
    """
    latest: dict[str, dict[str, Any]] = {}
    for run in sorted(check_runs, key=lambda r: r.get("started_at") or ""):
        name = run.get("name")
        if not isinstance(name, str) or name in SELF_NAMES:
            continue
        latest[name] = run
    return latest


def evaluate(
    check_runs: list[dict[str, Any]], required: set[str],
) -> dict[str, list[str]]:
    """Classify the required contexts. Advisory results are ignored.

    `missing` is required contexts with no check-run on the SHA at all. They
    are held separately from `pending` because they mean something different:
    pending is "reported, not finished", missing is "never reported", and only
    the second can sit there forever. The caller decides which is fatal.
    """
    latest = latest_per_name(check_runs)
    failing: list[str] = []
    pending: list[str] = []
    missing: list[str] = []
    for name in sorted(required):
        run = latest.get(name)
        if run is None:
            missing.append(name)
        elif run.get("status") in PENDING_STATUSES:
            pending.append(name)
        elif run.get("conclusion") in FAILING_CONCLUSIONS:
            failing.append(name)
    advisory_failing = sorted(
        name
        for name, run in latest.items()
        if name not in required and run.get("conclusion") in FAILING_CONCLUSIONS
    )
    return {
        "required": sorted(required),
        "failing": failing,
        "pending": pending,
        "missing": missing,
        "advisory_failing": advisory_failing,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollup", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True)
    args = parser.parse_args(argv)

    rollup = json.loads(args.rollup.read_text())
    check_runs = rollup.get("check_runs") if isinstance(rollup, dict) else rollup
    if not isinstance(check_runs, list):
        print("merge-train-gate: unreadable check-run payload", file=sys.stderr)
        return 2

    rules = json.loads(args.rules.read_text())
    if not isinstance(rules, list):
        print("merge-train-gate: unreadable branch-rules payload", file=sys.stderr)
        return 2

    required = required_contexts(rules)
    if not required:
        # Fail closed. An empty set here is indistinguishable from "the
        # ruleset moved" or "the token lost read access", and treating it
        # as "nothing is required" would merge anything.
        print(
            "merge-train-gate: resolved zero required contexts from the "
            "branch rules — refusing to decide. This is a fail-closed abort, "
            "not a green light.",
            file=sys.stderr,
        )
        return 2

    print(json.dumps(evaluate(check_runs, required), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
