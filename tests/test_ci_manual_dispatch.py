"""The workflows carrying required checks must be manually re-runnable (#1436).

Every required status context lives in `ci.yml` or `staging-gate.yml`. Both were
`on: pull_request` only, so when GitHub stopped delivering `pull_request`
webhooks there was no way to make a PR green — close/reopen, a label cycle and a
force-push of an amended commit all produced no run. `workflow_dispatch` is
delivered over the REST API rather than the webhook path, so it is the escape
hatch.

The interesting failure is not "the trigger is missing". It is a dispatch that
fires and reports a **pass having run nothing** — the #1160 shape. A dispatch
carries no `pull_request` payload, so:

* every `${{ github.event.pull_request.* }}` expression evaluates to the empty
  string, and a range like `"${BASE_SHA}..${HEAD_SHA}"` silently degenerates to
  `..` (an empty commit range) rather than failing; and
* `dorny/paths-filter` has no diff base, so a step gated on its output would be
  skipped and the job would report success from an `echo`.

These tests therefore assert the *degradation behaviour*, not just the presence
of the trigger, and they derive it from the workflow files rather than comparing
a list against itself (the tautological-guard failure of #1161).

Parsed by line rather than with PyYAML on purpose: `yaml` is not in this
project's dependency set — it reaches a local venv only as a transitive dep of
optional extras, so a `import yaml` here passes locally and fails under CI's
`uv sync --frozen --group dev --extra archive`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO / ".github" / "workflows"

# The five required contexts and the workflow each is defined in. This mapping
# is repository configuration that lives outside the tree (a branch-protection
# ruleset), so it is stated rather than derived — but it is only ever read to
# decide *which* files the invariants below apply to, never to assert itself.
_REQUIRED_CONTEXT_WORKFLOWS = {
    "ci.yml": ("pytest (3.12)", "pytest (3.13)"),
    "staging-gate.yml": ("secrets-scan", "pattern-scan", "history-scan"),
}

_PR_FIELD_RE = re.compile(r"github\.event\.pull_request\.[A-Za-z_.]+")
_EXPR_RE = re.compile(r"\$\{\{(.+?)\}\}")
_IF_RE = re.compile(r"^\s*(?:-\s+)?if:\s*(.+?)\s*$")


def _text(workflow: str) -> str:
    return (_WORKFLOWS / workflow).read_text(encoding="utf-8")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block(text: str, key: str, at_indent: int) -> list[str]:
    """Lines nested under `key:` — everything more-indented, comments dropped."""
    lines = text.splitlines()
    want = " " * at_indent + key + ":"
    try:
        start = next(i for i, line in enumerate(lines) if line.rstrip() == want)
    except StopIteration:  # pragma: no cover - guarded by callers' asserts
        return []
    out = []
    for line in lines[start + 1 :]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if _indent(line) <= at_indent:
            break
        out.append(line)
    return out


def _child_keys(block: list[str], at_indent: int) -> list[str]:
    return [
        line.strip().rstrip(":")
        for line in block
        if _indent(line) == at_indent and line.strip().endswith(":")
    ]


def _jobs(workflow: str) -> dict[str, str]:
    """Map job name -> the raw text of that job's body."""
    block = _block(_text(workflow), "jobs", 0)
    jobs: dict[str, str] = {}
    current: str | None = None
    for line in block:
        if _indent(line) == 2 and line.strip().endswith(":"):
            current = line.strip().rstrip(":")
            jobs[current] = ""
        elif current is not None:
            jobs[current] += line + "\n"
    return jobs


@pytest.mark.parametrize("workflow", sorted(_REQUIRED_CONTEXT_WORKFLOWS))
def test_required_check_workflow_accepts_manual_dispatch(workflow: str) -> None:
    """Without this, a PR whose events are not delivered can never go green."""
    triggers = _child_keys(_block(_text(workflow), "on", 0), 2)
    contexts = ", ".join(_REQUIRED_CONTEXT_WORKFLOWS[workflow])
    assert "workflow_dispatch" in triggers, (
        f"{workflow} carries required contexts ({contexts}) but its triggers are "
        f"{triggers} — no workflow_dispatch, so those checks cannot be re-run "
        "without a push. See #1436."
    )


@pytest.mark.parametrize("workflow", sorted(_REQUIRED_CONTEXT_WORKFLOWS))
def test_dispatch_declares_no_ref_input(workflow: str) -> None:
    """A `ref` input would turn the hatch into a way to skip the gate.

    Check-runs attach to the head SHA of the ref a run was dispatched *on*, and
    both branch protection and merge-train evaluate checks on the PR head SHA.
    So the built-in dispatch ref makes a run structurally unable to report
    against a commit it did not test. A `ref` input breaks that: checkout would
    read the input while the check-runs still landed on the dispatch ref.
    """
    nested = _block(_text(workflow), "workflow_dispatch", 2)
    assert nested == [], (
        f"{workflow}'s workflow_dispatch declares {nested!r}. Inputs that "
        "redirect what gets checked out let a run report against a commit it "
        "did not test (#1436 AC5). Target a branch with `--ref` instead."
    )


def _step_conditions(workflow: str) -> list[str]:
    return [m.group(1) for m in map(_IF_RE.match, _text(workflow).splitlines()) if m]


def test_every_paths_filter_gated_step_admits_a_dispatch() -> None:
    """A dispatch has no diff base, so filter-gated steps must still run.

    If a step is gated on `steps.filter.outputs.code` and does not also admit
    `workflow_dispatch`, a dispatched run skips it and the job reports a pass
    having executed nothing — a required check that is green and empty, which is
    exactly the hole #1160 closed on the docs-only path.
    """
    gated = [
        cond
        for cond in _step_conditions("ci.yml")
        if "steps.filter.outputs.code == 'true'" in cond
    ]
    assert gated, "expected ci.yml to still gate its install/test steps on the filter"
    for cond in gated:
        # The token alone is not enough, and the difference is the whole test:
        # `… && github.event_name != 'workflow_dispatch'` also contains the
        # word, and it skips every setup and test step on a dispatch while the
        # docs-only branch stays skipped too — so the job reports success having
        # run nothing. Require the positive equality, joined as a disjunct.
        assert "github.event_name == 'workflow_dispatch'" in cond, (
            "ci.yml step gated on the paths-filter without a positive "
            f"workflow_dispatch disjunct, so a dispatched run would skip it: {cond!r}"
        )
        assert "||" in cond, (
            "the workflow_dispatch condition must be a disjunct — joined with "
            f"`&&` it narrows the gate instead of widening it: {cond!r}"
        )


def test_the_docs_only_shortcut_does_not_swallow_a_dispatch() -> None:
    """The `echo` branch reports success without running tests.

    It is correct for a docs-only PR and wrong for a dispatch, which is fired
    precisely because someone needs the suite to actually execute.
    """
    shortcuts = [
        cond
        for cond in _step_conditions("ci.yml")
        if "steps.filter.outputs.code != 'true'" in cond
    ]
    assert shortcuts, "expected ci.yml to keep the docs-only shortcut"
    for cond in shortcuts:
        assert "github.event_name != 'workflow_dispatch'" in cond, (
            "ci.yml's docs-only shortcut would fire on a manual dispatch and "
            f"report a pass having run nothing: {cond!r}"
        )


def test_the_paths_filter_itself_is_pull_request_only() -> None:
    """`dorny/paths-filter` has no diff base on a dispatch.

    Left ungated it resolves its base to the default branch, which is the right
    answer for a PR head and the wrong one for a dispatch on `main` — where the
    diff is empty and every gated step would be skipped.
    """
    lines = _text("ci.yml").splitlines()
    uses = next(
        i
        for i, line in enumerate(lines)
        if line.strip().startswith("uses:") and "dorny/paths-filter" in line
    )
    guard = lines[uses - 1]
    assert "github.event_name == 'pull_request'" in guard, (
        "the paths-filter step must be pull_request-only; found preceding line "
        f"{guard!r}"
    )


def _job_conditions(workflow: str) -> dict[str, str]:
    """Job name -> its job-level `if`, for jobs that have one."""
    out: dict[str, str] = {}
    for name, body in _jobs(workflow).items():
        for line in body.splitlines():
            m = re.match(r"^    if:\s*(.+?)\s*$", line)
            if m:
                out[name] = m.group(1)
                break
    return out


@pytest.mark.parametrize("workflow", sorted(_REQUIRED_CONTEXT_WORKFLOWS))
def test_pull_request_payload_reads_degrade_rather_than_empty_out(workflow: str) -> None:
    """On a dispatch these expressions are `""`. Each must handle that.

    Three resolutions are accepted, and nothing else:

    1. the *job* runs only on `pull_request`, so the field always exists;
    2. the expression carries a `||` fallback in the template; or
    3. the value is bound to an env var whose emptiness the job's own shell
       tests before use — `commit-msg-prefix` does this, reconstructing the
       range from a merge-base, because there is no template-level fallback for
       `base.sha`.

    A bare read with none of the three is the silent-empty-range bug: the check
    reports green having validated nothing.

    Resolution 1 is read off the job-level `if:` and nothing else. Matching the
    guard anywhere in the job body exempts a job for a *step*-level guard, which
    says nothing about the steps around it: ci.yml's `pytest` job gates only its
    `dorny/paths-filter` step on `github.event_name == 'pull_request'`, and that
    substring alone excused the entire job — a bare
    `${{ github.event.pull_request.base.sha }}` added to its test step left this
    file green.
    """
    conditions = _job_conditions(workflow)
    for name, body in _jobs(workflow).items():
        if "github.event_name == 'pull_request'" in conditions.get(name, ""):
            continue
        for line in body.splitlines():
            for expr in _EXPR_RE.findall(line):
                if not _PR_FIELD_RE.search(expr) or "||" in expr:
                    continue
                binding = re.match(r"\s*([A-Z_][A-Z0-9_]*):\s*\$\{\{", line)
                assert binding, (
                    f"{workflow} job {name!r} interpolates {expr!r} inline, so a "
                    "workflow_dispatch run substitutes the empty string with no "
                    f"way to detect it (#1436): {line.strip()!r}"
                )
                var = binding.group(1)
                assert re.search(rf'-z\s+"\$\{{{var}\}}"', body), (
                    f"{workflow} job {name!r} reads {expr!r} into ${var} with no "
                    "`||` fallback, no pull_request-only guard, and no emptiness "
                    "check in its shell. On a workflow_dispatch that value is the "
                    "empty string (#1436)."
                )


def test_the_degradation_invariant_is_not_vacuous() -> None:
    """`staging-gate.yml` must still contain the case the invariant is about.

    Its three required scan jobs read the PR payload for their diff base. If
    none of them does any more, the parametrised test above passes by having
    nothing to check — so pin that at least one un-defaulted read survives and
    is therefore actually being exercised.
    """
    conditions = _job_conditions("staging-gate.yml")
    bare = [
        expr
        for name, body in _jobs("staging-gate.yml").items()
        if "github.event_name == 'pull_request'" not in conditions.get(name, "")
        for expr in _EXPR_RE.findall(body)
        if _PR_FIELD_RE.search(expr) and "||" not in expr
    ]
    assert bare, (
        "no un-defaulted pull_request read left in staging-gate.yml — if that is "
        "deliberate, delete the degradation test rather than letting it pass "
        "vacuously"
    )


@pytest.mark.parametrize("workflow", sorted(_REQUIRED_CONTEXT_WORKFLOWS))
def test_no_job_here_is_guarded_to_pull_request_only(workflow: str) -> None:
    """A guarded job does not vanish on a dispatch — it reports `skipped`.

    And `skipped` is the dangerous conclusion, not a neutral one. Merge-train
    evaluates `latest_per_name`, keeping the newest run per check name, and
    `skipped` is not in `FAILING_CONCLUSIONS` — so a dispatch's skipped row
    lands on the same head SHA with a later `started_at` and **overwrites an
    earlier real `failure`**. The gate flips green and the train merges.

    That turns the escape hatch into a way to clear a red check, which is the
    one thing #1436 AC5 forbids. The two jobs that genuinely cannot run outside
    a pull request live in `pr-metadata.yml`, which has no `workflow_dispatch`,
    so they cannot produce a row on a dispatch at all.
    """
    for name, cond in _job_conditions(workflow).items():
        assert "github.event_name == 'pull_request'" not in cond, (
            f"{workflow} job {name!r} is guarded to pull_request only, so a "
            "dispatched run emits a `skipped` check-run under that name — which "
            "supersedes an earlier failure in merge-train's per-name latest "
            f"rollup and clears the gate (#1436 AC5). Guard: {cond!r}"
        )


@pytest.mark.parametrize("workflow", sorted(_REQUIRED_CONTEXT_WORKFLOWS))
def test_checkout_never_pins_a_ref_in_these_workflows(workflow: str) -> None:
    """`ref:` on checkout is the back door the missing `ref` input leaves open.

    `test_dispatch_declares_no_ref_input` closes the front one. `actions/checkout`
    with `ref: main` makes the run *test* one commit while its check-runs land on
    the head SHA of the ref it was dispatched on — literally the "checkout would
    test one ref while the check-runs landed on another" failure the design
    comment claims is structurally impossible.
    """
    lines = _text(workflow).splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "actions/checkout" not in stripped or "uses:" not in stripped:
            continue
        step_indent = _indent(line)
        for follow in lines[i + 1 :]:
            if not follow.strip():
                continue
            if _indent(follow) <= step_indent:
                break
            assert not re.match(r"\s*ref:\s", follow), (
                f"{workflow}: actions/checkout at line {i + 1} pins a ref "
                f"({follow.strip()!r}). On a dispatch the run would test that ref "
                "while reporting its check-runs against the dispatched ref's head "
                "SHA (#1436 AC5)."
            )


def test_the_gated_step_still_runs_the_suite() -> None:
    """The dispatch guarantee is worth nothing if the step stops testing.

    Every other assertion here is about *whether* the step runs. This one is
    about whether running it means anything — replacing the command with an
    `echo` satisfies all of them.
    """
    runs = [
        line.strip()
        for line in _text("ci.yml").splitlines()
        if line.strip().startswith("run:")
    ]
    assert any("pytest tests/" in r and "--ignore=tests/e2e" in r for r in runs), (
        f"ci.yml no longer runs the suite on the gated path: {runs!r}"
    )


def test_the_empty_range_is_reconstructed_not_merely_detected() -> None:
    """`commit-msg-prefix` must rebuild the range, not just notice it is empty.

    The degradation test greps for a `-z "${VAR}"` emptiness test, which a branch
    that detects the empty case and then does nothing useful also satisfies —
    `BASE_SHA="${HEAD_SHA}"` inside the guard passes it and restores the empty
    commit range the whole file exists to prevent. Require the reconstruction.
    """
    body = _jobs("staging-gate.yml")["commit-msg-prefix"]
    assert re.search(r'BASE_SHA="\$\(git merge-base ', body), (
        "commit-msg-prefix's empty-payload branch must reconstruct the range "
        "from a merge-base against the default branch; detecting the empty case "
        "without rebuilding it validates nothing and exits 0 (#1436)."
    )


def test_the_pr_metadata_workflow_cannot_be_dispatched() -> None:
    """The two PR-only jobs are only safe while nothing can dispatch them.

    They moved out of `staging-gate.yml` precisely so a dispatch cannot emit a
    `skipped` row under their names. Adding `workflow_dispatch` here would put
    the masking mechanism straight back.
    """
    triggers = _child_keys(_block(_text("pr-metadata.yml"), "on", 0), 2)
    assert triggers == ["pull_request"], (
        "pr-metadata.yml must stay pull_request-only — a dispatchable run of "
        "these jobs emits `skipped` check-runs that supersede an earlier "
        f"failure in merge-train's rollup (#1436 AC5). Triggers: {triggers}"
    )
    for job in ("pr-title-prefix", "pr-body-issue-link"):
        assert job in _jobs("pr-metadata.yml"), f"{job} left pr-metadata.yml"
        assert job not in _jobs("staging-gate.yml"), (
            f"{job} is back in staging-gate.yml, which is dispatchable"
        )


# Keys that take the SAME value on a `pull_request` run and on a
# `workflow_dispatch` run of the same branch. Any of them in the concurrency
# group makes the two events collide.
_COLLIDING_CONCURRENCY_KEYS = ("github.head_ref", "github.ref_name", "github.sha")


@pytest.mark.parametrize("workflow", sorted(_REQUIRED_CONTEXT_WORKFLOWS))
def test_a_dispatch_cannot_cancel_a_pull_requests_own_run(workflow: str) -> None:
    """`cancel-in-progress` plus a colliding group key would cancel a required check.

    And a cancelled check-run reads as **green**, not as a failure: merge-train's
    `FAILING_CONCLUSIONS` is `{failure, timed_out, action_required}`, a completed
    run is not in `PENDING_STATUSES`, and the row exists so it is not `missing`
    either. Every signal is an absence test that a cancelled row satisfies — the
    same shape as the empty-rollup hole #1435 closed.

    The group is safe today because its two arms cannot produce the same value:
    a `pull_request` run keys on `github.event.pull_request.number` (an integer)
    and a dispatch falls through to `github.ref` (`refs/heads/<branch>`). Keys
    like `github.head_ref` or `github.ref_name` are identical across the two
    events, so substituting one would make a dispatch cancel the PR's in-flight
    run and turn its required checks green-by-cancellation.
    """
    lines = _text(workflow).splitlines()
    i = next(k for k, line in enumerate(lines) if line.rstrip() == "concurrency:")
    group = next(l for l in lines[i + 1 :] if l.strip().startswith("group:"))
    cancels = any(
        "cancel-in-progress: true" in l for l in lines[i + 1 : i + 4]
    )
    if not cancels:
        return
    assert "github.event.pull_request.number" in group, (
        f"{workflow}: with cancel-in-progress, the concurrency group must key the "
        "pull_request arm on the PR number so it can never equal the dispatch "
        f"arm's `github.ref` (#1436): {group.strip()!r}"
    )
    for key in _COLLIDING_CONCURRENCY_KEYS:
        assert key not in group, (
            f"{workflow}: `{key}` takes the same value on a pull_request run and "
            "on a dispatch of the same branch, so with cancel-in-progress a "
            "dispatch would cancel the PR's in-flight run — and a cancelled "
            "required check-run evaluates as green in merge-train's rollup "
            f"(#1436): {group.strip()!r}"
        )
