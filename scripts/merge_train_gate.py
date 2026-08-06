"""Decide whether the merge-train may proceed, from check-runs (#1397).

The gate used to be inline `jq` in `.github/workflows/merge-train.yml`, and it
enumerated **every** check-run on the head SHA. That made `Sourcery review`,
CodeRabbit and every other advisory bot de facto merge-blocking, while the
failure message said `required check(s) failed` — naming a set the workflow
never read. PR #1394 sat blocked behind a verified-false SQL-injection finding
because of it.

This module keeps gating on **every** check-run and excludes advisory bots by
name. Gating on the *required set only* was tried and rejected in review: the
required set is 5 contexts while a PR carries ~25 check-runs, so required-only
would demote **19** real gates — including `migration-policy-check`, which
exists because a migration once collided the `edges` primary key and left
stores unopenable forever, and `release-docs-check`, which carries the
CHANGELOG-duplicate detector. Promoting them into the required set is not an
option either: a path-filtered check that does not run on a given PR would sit
permanently pending and brick it, which is the documented reason the
replay-soak gate was never made required.

So this trades the narrow fix for the narrow problem. One name comes out of the
gate; nothing else is demoted.

The required set is still resolved at run time, but only to **label** which
failures were required — the message misattributing an advisory bot as required
was the second half of #1397.

It lives here rather than in the workflow because a gate that cannot be tested
is how the original defect survived. The decision is a pure function of
(rollup, required set) and is exercised directly by
`tests/test_merge_train_gate.py`, including the case that matters most — a red
advisory check with every required context green must **merge**.

**An unresolvable required set degrades the message, not the decision.** Since
the gate covers every non-advisory check either way, the required set never
narrows it — losing the set costs only the ability to *label* which failures
were required. So resolution returning nothing warns and continues rather than
aborting: unknown means gate on more, never on less. (Under the required-only
design this file no longer implements, an empty set would have meant "nothing
is required, merge anything", and aborting would have been the only safe
reading. That is not the trade this code makes.)

Two behaviours are inherited deliberately and must not be simplified away:

* **Per-name latest-run dedup (#632).** A superseded run leaves a `cancelled`
  row on the same SHA; keeping only the newest row per name is what stops it
  blocking merge. `cancelled` is therefore *not* a failure — after the dedup a
  genuine cancellation surfaces as the only row for its name and an operator
  can re-trigger.
* **`pending` is scoped to the same non-advisory set as `failing`**, so a slow
  or silent *advisory* bot no longer holds the train to
  `CHECK_TIMEOUT_SECONDS`. Any other slow check still does, by design — it
  gates, so waiting for it is the point.

Usage::

    gh api --paginate "repos/${REPO}/commits/${SHA}/check-runs?per_page=100" \
        | jq -s '{check_runs: [.[].check_runs[]]}' > rollup.json
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

# Advisory bots: reviewers whose opinion is worth reading and must not decide
# whether `main` moves. This is a literal list because no repo API distinguishes
# "advisory" from "gating" — the ruleset only knows *required*, and required is
# the wrong axis (see the module docstring).
#
# A name that stops matching is the failure mode, so the gate reports which
# entries matched nothing on this SHA rather than letting a rename silently
# restore blocking. `CodeRabbit` posts a commit *status* rather than a
# check-run, so it never reached this filter and its entry is inert today —
# kept because it costs nothing and a bot can change surface.
ADVISORY_NAMES: frozenset[str] = frozenset({"Sourcery review", "CodeRabbit"})

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
    """Classify every check-run. Advisory names are excluded from the gate.

    `failing` and `pending` cover everything that is not this train's own job
    and not advisory — so a red `migration-policy-check` still blocks, exactly
    as before. `required` is used only to annotate which of the failures were
    required contexts; it never narrows the gate.

    `missing` is a required context with no check-run on the SHA at all, held
    separately from `pending` because they mean different things: pending is
    "reported, not finished", missing is "never reported", and only the second
    can sit forever. It is reported, not gated on, since a required context
    that never posts is already fatal at push time.
    """
    latest = latest_per_name(check_runs)
    gating = {n: r for n, r in latest.items() if n not in ADVISORY_NAMES}

    failing = sorted(
        n for n, r in gating.items()
        if r.get("conclusion") in FAILING_CONCLUSIONS
    )
    pending = sorted(
        n for n, r in gating.items() if r.get("status") in PENDING_STATUSES
    )
    advisory_failing = sorted(
        n for n, r in latest.items()
        if n in ADVISORY_NAMES and r.get("conclusion") in FAILING_CONCLUSIONS
    )
    return {
        "required": sorted(required),
        "failing": failing,
        "failing_required": sorted(n for n in failing if n in required),
        "failing_not_required": sorted(n for n in failing if n not in required),
        "pending": pending,
        "missing": sorted(n for n in required if n not in latest),
        "advisory_failing": advisory_failing,
        # Advisory entries that matched nothing on this SHA. A renamed bot
        # would silently start blocking again, so the run says so out loud.
        "advisory_unmatched": sorted(ADVISORY_NAMES - set(latest)),
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

    # An unresolvable required set no longer narrows anything — the gate
    # covers every non-advisory check either way — so it degrades the
    # *message* rather than the decision. That is the fail-closed direction:
    # unknown means gate on more, never on less.
    required = required_contexts(rules)
    if not required:
        print(
            "merge-train-gate: resolved zero required contexts from the "
            "branch rules; gating is unaffected (every non-advisory check "
            "still gates), but failures cannot be labelled required.",
            file=sys.stderr,
        )

    print(json.dumps(evaluate(check_runs, required), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
