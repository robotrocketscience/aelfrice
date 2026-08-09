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
        assert "workflow_dispatch" in cond, (
            "ci.yml step gated on the paths-filter without a workflow_dispatch "
            f"disjunct, so a dispatched run would skip it: {cond!r}"
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


@pytest.mark.parametrize("workflow", sorted(_REQUIRED_CONTEXT_WORKFLOWS))
def test_pull_request_payload_reads_degrade_rather_than_empty_out(workflow: str) -> None:
    """On a dispatch these expressions are `""`. Each must handle that.

    Three resolutions are accepted, and nothing else:

    1. the job runs only on `pull_request`, so the field always exists;
    2. the expression carries a `||` fallback in the template; or
    3. the value is bound to an env var whose emptiness the job's own shell
       tests before use — `commit-msg-prefix` does this, reconstructing the
       range from a merge-base, because there is no template-level fallback for
       `base.sha`.

    A bare read with none of the three is the silent-empty-range bug: the check
    reports green having validated nothing.
    """
    for name, body in _jobs(workflow).items():
        if "github.event_name == 'pull_request'" in body:
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
    bare = [
        expr
        for body in _jobs("staging-gate.yml").values()
        if "github.event_name == 'pull_request'" not in body
        for expr in _EXPR_RE.findall(body)
        if _PR_FIELD_RE.search(expr) and "||" not in expr
    ]
    assert bare, (
        "no un-defaulted pull_request read left in staging-gate.yml — if that is "
        "deliberate, delete the degradation test rather than letting it pass "
        "vacuously"
    )
