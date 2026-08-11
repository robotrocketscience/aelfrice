"""Post-merge workflow runs must actually happen again (#1423).

`merge-train.yml` lands every merge as
`git push origin "${HEAD_SHA}:refs/heads/main"` with `secrets.GITHUB_TOKEN`.
GitHub raises no workflow runs from events made with that token — the
documented recursion guard, whose only exceptions are `workflow_dispatch` and
`repository_dispatch`. So every `on: push: branches: [main]` workflow in this
repo stopped on 2026-07-21 and `main` moved 942 commits with none of them
firing. Two of them (`release-drafter`, `flag-stale-open-prs`) have no second
trigger, so they were not late — they were off.

The interesting failure mode is not "the dispatch is missing". It is a
dispatch that exists and covers the wrong set, or fires at the wrong moment:

* a **hand-written list** of workflows to dispatch omits the next `push:main`
  workflow someone adds, silently, which is the original defect wearing a
  different hat. Hence a derived list, and hence the assertion below is that
  nothing enumerable is written down rather than that a particular name is;
* a dispatch **before** the FF push targets `main` at its pre-merge SHA, which
  produces a green post-merge run for a commit that never merged;
* a dispatch with **`actions: write` missing** 403s, and the `|| warning` that
  keeps a failed dispatch from failing a landed merge swallows it — the fix
  reverts itself and the logs say so once, in a warning, weekly, unread;
* a `push`-gated **failure-surfacing** step does not fire on a dispatch, so a
  red post-merge e2e opens no issue and tells nobody. That step is the entire
  reason (#370) the post-merge run exists.

Parsed by line rather than with PyYAML on purpose: `yaml` is not in this
project's dependency set — it reaches a local venv only as a transitive dep of
optional extras, so `import yaml` here passes locally and fails under CI's
`uv sync --frozen --group dev --extra archive`. Same trap as
`tests/test_ci_manual_dispatch.py`, whose helpers this file reuses in shape.
"""

from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

# Every test here that spawns the enumeration script carries its own ceiling.
# The suite default is 5s, sized for a test that does no I/O, and a process
# spawn under contention reports as a hang rather than as slowness on it
# (#1307). The same number bounds the child itself, so neither layer can wait
# forever on the other.
_SPAWN_TIMEOUT_S = 30

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO / ".github" / "workflows"
_SCRIPT = _REPO / "scripts" / "push_trigger_workflows.py"
_MERGE_TRAIN = "merge-train.yml"
_HEARTBEAT = "push-trigger-heartbeat.yml"

_FF_PUSH = 'git push origin "${HEAD_SHA}:refs/heads/main"'


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
    except StopIteration:
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


def _code_lines(workflow: str) -> list[str]:
    """Workflow lines with comment-only lines removed.

    Prose in a comment may name a workflow file — that is documentation, and
    the drift risk this file is about is a name the *code* depends on.
    """
    return [
        line
        for line in _text(workflow).splitlines()
        if not line.lstrip().startswith("#")
    ]


# The script is exercised through its CLI rather than imported, because the
# CLI *is* the interface: `merge-train.yml` and the heartbeat both read its
# stdout. Cached because the parametrised tests below call it at collection
# time and once per case, and each call is a process spawn.
@lru_cache(maxsize=1)
def _script_output() -> tuple[str, ...]:
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--branch", "main"],
        capture_output=True,
        text=True,
        check=True,
        timeout=_SPAWN_TIMEOUT_S,
    )
    return tuple(proc.stdout.split())


def _push_main_workflows_independently() -> set[str]:
    """A second, deliberately different parse of the same question.

    Comparing the script against itself would be the tautological guard that
    made #1161's check worthless. This walks the files with its own state
    machine so agreement means something.
    """
    found = set()
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        in_on = False
        in_push = False
        in_branches = False
        hit = False
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            col = _indent(raw)
            stripped = raw.strip()
            if col == 0:
                in_on = stripped in ("on:", '"on":', "'on':")
                in_push = in_branches = False
                continue
            if not in_on:
                continue
            if col == 2:
                in_push = stripped == "push:"
                in_branches = False
                continue
            if not in_push:
                continue
            if col == 4:
                in_branches = stripped.startswith("branches:")
                if in_branches and "main" in stripped.replace("branches:", ""):
                    hit = True
                continue
            if in_branches and col > 4 and stripped in ("- main", "- 'main'", '- "main"'):
                hit = True
        if hit:
            found.add(path.name)
    return found


# --------------------------------------------------------------------------
# The enumeration itself
# --------------------------------------------------------------------------


@pytest.mark.timeout(_SPAWN_TIMEOUT_S)
def test_the_script_finds_every_push_main_workflow() -> None:
    """Independent parse, same answer — or the derived list is not derived."""
    assert set(_script_output()) == _push_main_workflows_independently()


@pytest.mark.timeout(_SPAWN_TIMEOUT_S)
def test_the_enumeration_is_not_empty_and_excludes_tag_only_pushes() -> None:
    """Both halves are load-bearing.

    An empty list makes every assertion below pass vacuously and makes the
    merge train dispatch nothing. And `publish.yml` is the discrimination that
    proves the parse reads `branches:` rather than `push:`: it pushes on
    `v*` **tags**, which the merge train never creates and which the
    `GITHUB_TOKEN` guard does not touch — dispatching it on every merge would
    attempt a release.
    """
    names = _script_output()
    assert len(names) >= 2, f"suspiciously small push:main enumeration: {names}"
    assert "publish.yml" not in names, (
        "publish.yml triggers on tag pushes, not `branches: [main]` — "
        "dispatching it after every merge would fire the release job"
    )


@pytest.mark.timeout(_SPAWN_TIMEOUT_S)
@pytest.mark.parametrize("workflow", _script_output())
def test_every_enumerated_workflow_accepts_a_dispatch(workflow: str) -> None:
    """`gh workflow run` 422s on a workflow with no `workflow_dispatch`.

    And merge-train's dispatch loop is non-fatal, so the 422 would surface as
    one warning line in a job whose overall conclusion is success.
    """
    triggers = _child_keys(_block(_text(workflow), "on", 0), 2)
    assert "workflow_dispatch" in triggers, (
        f"{workflow} declares `push: branches: [main]`, which cannot fire under "
        f"the merge train, but has no workflow_dispatch hatch: {triggers}. "
        "merge-train would try to dispatch it and get a 422 (#1423)."
    )


@pytest.mark.timeout(_SPAWN_TIMEOUT_S)
@pytest.mark.parametrize("workflow", [_MERGE_TRAIN, _HEARTBEAT])
def test_neither_consumer_writes_the_list_down(workflow: str) -> None:
    """A literal list is the defect, not the fix.

    Both consumers shell out to `scripts/push_trigger_workflows.py`. If either
    grew its own copy of the names, the next `push:main` workflow added would
    be dispatched-but-not-monitored or monitored-but-not-dispatched, and
    nothing would say so.
    """
    code = "\n".join(_code_lines(workflow))
    assert "push_trigger_workflows.py" in code, (
        f"{workflow} no longer derives its workflow list from the shared script"
    )
    for name in _script_output():
        assert name not in code, (
            f"{workflow} names {name!r} in a non-comment line. The list must be "
            "derived, or the next push:main workflow someone adds is silently "
            "left out — which is exactly how #1423 stayed invisible."
        )


# --------------------------------------------------------------------------
# merge-train's half
# --------------------------------------------------------------------------


def test_merge_train_may_dispatch_at_all() -> None:
    """Without `actions: write` the dispatch 403s and the warning hides it."""
    permissions = _block(_text(_MERGE_TRAIN), "permissions", 0)
    granted = {
        line.split(":")[0].strip(): line.split(":", 1)[1].split("#")[0].strip()
        for line in permissions
    }
    assert granted.get("actions") == "write", (
        "merge-train.yml needs `actions: write` to call `gh workflow run`; a "
        f"403 is swallowed by the non-fatal dispatch loop. Granted: {granted}"
    )


def _dispatch_line_numbers() -> list[int]:
    """Lines that *invoke* `gh workflow run`, not lines that mention it.

    Two things in this file mention it without running it: the inline comment
    on the `actions: write` grant, and `fail_and_unlabel`'s heredoc, which
    tells a human to `gh workflow run ci.yml --ref <branch>` when a required
    check never reported (#1436). Matching either made the assertions below
    pass against prose — and the refusal-path assertion fail against it.

    An invocation begins the command, optionally under an `if`.
    """
    return [
        i
        for i, line in enumerate(_text(_MERGE_TRAIN).splitlines())
        if line.strip().startswith(("gh workflow run", "if gh workflow run"))
    ]


def test_the_dispatch_happens_after_the_ff_push() -> None:
    """Dispatching first would test `main` at its pre-merge SHA.

    That is worse than not dispatching: it produces a green post-merge run
    attributed to a commit that had not landed, so the record says the merge
    was verified when what was verified is the state before it.
    """
    lines = _text(_MERGE_TRAIN).splitlines()
    push_at = next(i for i, line in enumerate(lines) if _FF_PUSH in line)
    dispatches = _dispatch_line_numbers()
    assert dispatches, "merge-train.yml no longer dispatches anything (#1423)"
    for at in dispatches:
        assert at > push_at, (
            f"`gh workflow run` at line {at + 1} precedes the FF push at line "
            f"{push_at + 1}, so it would dispatch `main` at its pre-merge SHA"
        )


def test_the_dispatch_is_not_inside_the_refusal_path() -> None:
    """`fail_and_unlabel` runs when the merge did *not* happen.

    It ends in `exit 0`, so a dispatch placed in its body would fire on every
    refusal and produce post-merge runs for merges that were rejected.
    """
    lines = _text(_MERGE_TRAIN).splitlines()
    start = next(i for i, line in enumerate(lines) if "fail_and_unlabel()" in line)
    end = next(
        i
        for i, line in enumerate(lines)
        if i > start and line.strip() == "}"
    )
    for at in _dispatch_line_numbers():
        assert not (start < at < end), (
            f"`gh workflow run` at line {at + 1} is inside fail_and_unlabel "
            f"(lines {start + 1}-{end + 1}), which runs when nothing merged"
        )


def test_the_dispatch_targets_main_and_carries_no_inputs() -> None:
    """The ref is the safety property (#1436/#1451), and inputs would break it.

    A dispatched run's check-runs attach to the head of the ref it was
    dispatched *on*. Targeting `main` is therefore what makes a post-merge run
    structurally unable to report against a commit it did not test. An input
    that redirects the checkout re-opens exactly that gap.
    """
    line = next(
        _text(_MERGE_TRAIN).splitlines()[at] for at in _dispatch_line_numbers()
    )
    assert "--ref main" in line, (
        f"the dispatch must name the branch it verifies: {line.strip()!r}"
    )
    for flag in (" -f ", "--field", "--raw-field", "--json"):
        assert flag not in line, (
            f"the dispatch passes {flag.strip()!r}; inputs let a run report "
            f"against a commit it did not test (#1436 AC5): {line.strip()!r}"
        )


def test_a_failed_dispatch_cannot_fail_a_landed_merge() -> None:
    """`main` has already moved by then; there is nothing left to abort.

    Turning the train red after a successful FF would leave a merged PR
    labelled as a failed merge and an operator re-adding `ready-to-merge` to a
    branch that is already in.
    """
    body = _text(_MERGE_TRAIN)
    at = _dispatch_line_numbers()[0]
    window = "\n".join(body.splitlines()[at - 1 : at + 6])
    assert "if gh workflow run" in window or "||" in window, (
        "the dispatch must be guarded so a failure warns rather than fails the "
        f"job: {window!r}"
    )
    assert "::warning::" in window, (
        "a swallowed dispatch failure must at least annotate the run: "
        f"{window!r}"
    )


# --------------------------------------------------------------------------
# What the dispatched runs must still do
# --------------------------------------------------------------------------


def test_a_dispatched_e2e_failure_still_opens_an_issue() -> None:
    """#370's failure surfacing was gated on `github.event_name == 'push'`.

    The post-merge run now arrives as `workflow_dispatch`, so that guard would
    skip the issue-opening step on every run it exists for — a red e2e on
    `main` with no issue, no label, and a job conclusion nobody is watching.
    """
    lines = _text("e2e.yml").splitlines()
    at = next(
        i
        for i, line in enumerate(lines)
        if "gh issue create" in line and not line.lstrip().startswith("#")
    )
    guard = next(
        line
        for line in reversed(lines[:at])
        if line.strip().startswith("if:") and "github.event_name" in line
    )
    assert "!= 'pull_request'" in guard or "workflow_dispatch" in guard, (
        "e2e.yml's failure-surfacing step must admit the dispatched post-merge "
        f"run, not only `push` (#1423): {guard.strip()!r}"
    )


# --------------------------------------------------------------------------
# AC4 — the regression detector
# --------------------------------------------------------------------------


def test_the_heartbeat_runs_on_a_schedule() -> None:
    """A detector that only runs when someone remembers is the status quo."""
    triggers = _child_keys(_block(_text(_HEARTBEAT), "on", 0), 2)
    assert "schedule" in triggers, (
        f"{_HEARTBEAT} must fire unattended: {triggers}"
    )
    permissions = _block(_text(_HEARTBEAT), "permissions", 0)
    granted = {line.split(":")[0].strip() for line in permissions}
    assert {"actions", "issues"} <= granted, (
        f"the heartbeat needs to read runs and open an issue: {granted}"
    )


def test_the_heartbeat_ignores_schedule_and_pull_request_runs() -> None:
    """Those are the masking, not the signal.

    `codeql.yml` and `zizmor.yml` both carry a weekly cron, so their newest run
    on `main` stayed recent through the entire outage. A heartbeat that counted
    it would have reported them healthy for all 942 of them.
    """
    code = "\n".join(_code_lines(_HEARTBEAT))
    assert "event=${ev}" in code, (
        f"the heartbeat must filter its freshness query by event: {code!r}"
    )
    # Matched exactly, not by containment: `for ev in push workflow_dispatch
    # schedule` contains the two-event substring and re-admits the masking.
    loop = next(line for line in _code_lines(_HEARTBEAT) if "for ev in " in line)
    assert loop.strip() == "for ev in push workflow_dispatch; do", (
        "the heartbeat's freshness query must cover exactly the two events that "
        f"represent a post-merge run: {loop.strip()!r}"
    )


def test_the_heartbeat_measures_lag_against_main_not_wall_clock() -> None:
    """Otherwise a quiet fortnight opens an issue about nothing.

    The defect is "`main` moved and the workflow did not follow". Comparing
    against `now` reports a repository with no merges as broken, which is how a
    detector gets muted and then ignored.
    """
    code = "\n".join(_code_lines(_HEARTBEAT))
    assert "repos/${REPO}/commits/main" in code, (
        "the heartbeat must resolve main's head commit date to compare against"
    )
    assert "(head_ts - run_ts)" in code, (
        "staleness must be measured as head-commit minus last-run, not against "
        f"wall-clock now: {code!r}"
    )


def test_the_heartbeat_refuses_an_empty_enumeration() -> None:
    """Zero workflows to check is a pass, and a pass is what it must not be."""
    code = "\n".join(_code_lines(_HEARTBEAT))
    assert 'if [ -z "${workflows}" ]' in code and "::error::" in code, (
        "an empty enumeration must fail the heartbeat rather than report every "
        f"workflow healthy: {code!r}"
    )


# --------------------------------------------------------------------------
# The parser, against files it has never seen
# --------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, body: str) -> Path:
    (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def _run(tmp_path: Path, branch: str = "main") -> list[str]:
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--workflows-dir",
            str(tmp_path),
            "--branch",
            branch,
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=_SPAWN_TIMEOUT_S,
    )
    return proc.stdout.split()


@pytest.mark.timeout(_SPAWN_TIMEOUT_S)
def test_parser_reads_flow_and_block_branch_lists(tmp_path: Path) -> None:
    _write(tmp_path, "flow.yml", "on:\n  push:\n    branches: [main, dev]\n")
    _write(
        tmp_path,
        "block.yml",
        "on:\n  push:\n    branches:\n      - 'main'\n      - dev\n",
    )
    assert _run(tmp_path) == ["block.yml", "flow.yml"]


@pytest.mark.timeout(_SPAWN_TIMEOUT_S)
def test_parser_ignores_tag_only_and_branch_ignore_pushes(tmp_path: Path) -> None:
    """`tags:` names a tag even when the tag is spelled like the branch.

    `publish.yml`'s real tag pattern is `v[0-9]+…`, which no branch-name match
    can reach, so a parser that read `tags:` as `branches:` would look correct
    against this repo and dispatch the release job the first time anyone cut a
    tag named after a branch. The fixture is the discriminating case, not the
    realistic one.
    """
    _write(tmp_path, "tags.yml", "on:\n  push:\n    tags:\n      - 'main'\n")
    _write(tmp_path, "flow-tags.yml", "on:\n  push:\n    tags: [main]\n")
    _write(tmp_path, "ignore.yml", "on:\n  push:\n    branches-ignore: [main]\n")
    assert _run(tmp_path) == []


@pytest.mark.timeout(_SPAWN_TIMEOUT_S)
def test_parser_does_not_read_a_pull_request_branch_list(tmp_path: Path) -> None:
    """`pull_request: branches: [main]` is not a push trigger.

    Every workflow in this repo that filters PRs to `main` would otherwise be
    dispatched after each merge — including required-context workflows, whose
    dispatched runs land check-runs on `main`'s head SHA.
    """
    _write(
        tmp_path,
        "pr.yml",
        "on:\n  pull_request:\n    branches: [main]\n\njobs:\n  a:\n    runs-on: x\n",
    )
    assert _run(tmp_path) == []


@pytest.mark.timeout(_SPAWN_TIMEOUT_S)
def test_parser_does_not_match_a_push_key_outside_the_on_block(tmp_path: Path) -> None:
    """A job or step named `push` must not be read as a trigger."""
    _write(
        tmp_path,
        "job.yml",
        "on:\n  pull_request:\n\njobs:\n  push:\n    branches: [main]\n",
    )
    assert _run(tmp_path) == []


@pytest.mark.timeout(_SPAWN_TIMEOUT_S)
def test_parser_accepts_the_quoted_on_key(tmp_path: Path) -> None:
    """Bare `on` is YAML 1.1's boolean `true`, so the quoted spelling is legal."""
    _write(tmp_path, "quoted.yml", '"on":\n  push:\n    branches: [main]\n')
    assert _run(tmp_path) == ["quoted.yml"]


@pytest.mark.timeout(_SPAWN_TIMEOUT_S)
def test_parser_matches_the_branch_literally(tmp_path: Path) -> None:
    """A wildcard is a deliberate superset, not a subscription to one branch.

    `maintenance` is the arm that discriminates: a substring test matches it
    against `main` and would dispatch, after every merge to `main`, a workflow
    that asked for a different branch entirely.
    """
    _write(tmp_path, "glob.yml", "on:\n  push:\n    branches: ['release/*']\n")
    _write(tmp_path, "sub.yml", "on:\n  push:\n    branches: [maintenance]\n")
    assert _run(tmp_path) == []
    assert _run(tmp_path, branch="release/*") == ["glob.yml"]
    assert _run(tmp_path, branch="maintenance") == ["sub.yml"]
