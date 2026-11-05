#!/usr/bin/env python3
"""Bisect a nightly canary regression to the merge that caused it (#1786).

The nightly canary runs the full suite on main. When a test that passed on
the last green commit fails tonight, and still passes when it runs on that
commit again, the cause is a commit between the two. This script finds it.

Usage:
    nightly_bisect.py merge-points --good SHA --bad SHA --out POINTS.json
        [--repo OWNER/NAME] [--repo-root DIR] [--dry-run]
    nightly_bisect.py run --good SHA --bad SHA (--tests IDS.txt | --triage TRIAGE.json)
        --out RESULT.json [--points POINTS.json] [--repo-root DIR] [--worktree DIR]
        [--budget-seconds 1200] [--step-timeout 300] [--step-cmd CMD]
        [--python VERSION] [--extra NAME ...] [--dry-run]
    nightly_bisect.py step --tests IDS.txt --deadline EPOCH
        [--step-timeout 300] [--failed-out FILE] [--absent-out FILE]
        [--python VERSION] [--extra NAME ...] [--dry-run]

`--dry-run` writes no file, calls no API and changes no git state. It
prints what the subcommand would do and exits as a real run would on
success.

Merge points. The merge train fast-forwards main, so a merged PR's merge
commit is its head commit, and the commits before it in the same PR were
never on main on their own. `merge-points` lists the commits on the
ancestry path from `--good` to `--bad` and keeps those that are the merge
commit of a PR merged since the day before `--good` was committed, as
`gh pr list --state merged` reports them. `--bad` is always kept. The
output maps each kept commit to its PR number, or to null for `--bad`
when no PR merged it, in history order. When the `gh` call fails, returns
as many PRs as its `--limit` (the list may be cut short), lists a PR with no
merge commit (the commit it merged cannot be told apart), or names no merge
in the range, the output is `{}` with a warning, and `run` then tests every
commit. `run` reads a points file with no PR in the range the same way,
since blaming `--bad` would test nothing.

A commit that no listed PR merged, a direct push for one, cannot always be
detected: it is skipped, and its break reads as the next merge point's. So
every culprit event also names `preceding`, the merge point it was checked
against, and an issue can say the test broke between the two.

The run needs one chain of commits, which main is: its last 710 first-parent
commits held no merge commit on 2026-11-04. The check on "the merge point
before the culprit" only means something on such a chain, so a range holding a
commit with more than one parent, or one off the ancestry path, tests nothing
and gets a range event with the reason `nonlinear`.

The run. `run` adds a detached worktree at `--good` (`.bisect` under the
repository root by default, removed first if a previous run left it), and
in it runs `git bisect start BAD GOOD`, marks every commit that is not a
merge point with `git bisect skip`, and runs `git bisect run` with the
step. The step is this script's `step` subcommand, invoked by its absolute
path, because `git bisect run` runs it inside the worktree, where
`scripts/` holds the bisected commit's files. `--step-cmd` replaces the
step command; the tests use it to run a fake in place of uv and pytest.

The result comes from `refs/bisect/bad` and `refs/bisect/good-*`, never
from the exit status of `git bisect run`, which is nonzero both when it
stops on skipped commits and when a step aborts it. The bad commit is the
culprit when the merge point before it is good. The checks below then run
the step outside the bisect. When the deadline stops a check, the range is
the one the bisect narrowed to.

`git bisect reset` runs in a `finally` after every round. A test that
dirties a tracked file can make it fail, so it runs again after `git reset
--hard`; a reset that still fails is a warning and lands in the result's
`reset_errors`. The worktree stays after the run; the next run removes it
first, including when its folder or its `.git` file is gone.

This script removes only a folder it owns: the default `.bisect`, or a folder
holding the `.nightly-bisect-worktree` file a run wrote when it created that
worktree. Any other `--worktree` that holds files, the repository root and
another worktree among them, exits 2 with nothing touched, in a dry run too.
A folder that is missing or empty is used. A symbolic link exits 2.

The checks, per test, at each culprit. The step runs on the culprit, which
has to fail, and on the merge point before it, which has to pass. A test
that fails at that merge point too gets a range event up to the culprit.
A status of 1 that names no test, as when a test kills pytest before it writes
its report, says nothing about any one test, so every test of that check gets
a range event over the round.
The tests that failed at the culprit then run there a second time, and only
a test that fails both times is blamed on it. One that passes the second
time flaked, and goes to the next round. A real break stays broken, so the
blamed tests also run at the merge point after the culprit, unless that is
`--bad`: one that passes there gets a range event.

A commit where a test does not exist reads good, so a test that existed
under another name, or was gone for a while, looks new at the commit that
brought it back. When a blamed test is absent at the merge point before the
culprit, and git sees its file renamed or copied between the two, or the
test is present at the round's first commit, or that commit cannot be judged,
it gets a range event up to the culprit. A test absent all the way back is
new, and is blamed on the commit that added it.

What the checks cannot detect:

- A test that gives the same wrong result at every run of a commit. A false
  good at the real culprit, at the bisect and again at the check, moves the
  blame to the next merge point, and that commit fails as a later commit
  should. No fixed number of reruns closes this. A false bad is caught only
  when the merge point after it passes, so not when the real break is that
  next point.
- A test function renamed inside its file, or a file renamed with too little
  left for git's similarity check: the new name reads as a new test.
- A test that breaks, recovers and breaks again. The bisect finds one change
  from pass to fail, not all of them.

Several culprits. The step fails when any of its tests fails, so the
first round finds the first commit that broke one of them. Tests that
still pass at the culprit broke later, and get a new round from the culprit
to `--bad`, up to three rounds within the budget. What is left after the
last round gets a range event.

`git bisect run` exports `GIT_DIR` to the step. A test that runs `git init`
in a temporary directory under that variable re-initializes the worktree's
git directory instead, so the step's children, and every git call here, run
without git's repository variables.

The step. `--failed-out` and `--absent-out` name the tests that failed and
the tests that do not exist at the commit. Its exit status follows `git
bisect run`: 0 good, 1 bad, 125 skip, 255 abort. Past `--deadline` it exits
255, which stops the bisect; the result then names the range it had
narrowed to. Otherwise it runs
`uv sync --frozen` (125 when that fails), collects the test files with
`pytest --collect-only -q` (125 when collection fails) and keeps the ids
it collected. When none is collected the commit predates the tests and is
good. It then runs those ids: pytest's 0 is good, 1 is bad, and any other
status or a timeout is a skip. A module that fails to import, or a folder
whose `conftest.py` does, reaches the step as a target (`tests.test_x`, the
name pytest's JUnit report gives a collection error, or `tests/sub/`), judged
by collecting it alone as below.
`--step-timeout` bounds the whole step. A step that cannot read its input
exits 255 rather than 2, which `git bisect run` would read as a bad commit, and
a bad flag or any other uncaught error exits 125 for the same reason.

Test ids. `--tests` takes three forms: a node id
(`tests/test_x.py::TestC::test_a[p]`), a JUnit id
(`tests.test_x.TestC::test_a[p]`, what `test_outcome_record.py` records),
and a target (`tests.test_x`, `::tests.test_x`, `tests/test_x.py`, `::tests.sub`
or `tests/sub/`, where a dotted id names the folder when no such module exists).
`run` maps each to a node id against the tree of `--bad`, where a test that
failed tonight has to exist, and the result lists the ids as given. An id
that names no file there gets a range event with the reason `unmapped`
without any step. Given a JUnit id directly, `step` maps it against the
checked-out tree and exits 125 when it names no file there, because it
cannot tell a test that does not exist yet from an id it cannot read. A
node id or target whose path is absent means the test does not exist at
that commit, which is good.

`--triage` reads what `nightly_canary.py regressions` writes, in place of
`--tests`: every test of `candidates` and `new_failing` by its `node_id`, and
every entry of `collection_errors` by its `module`. A null there is an id
that mapped to no file, and gets a range event as `unmapped`. The result
names each by its `test`, the id the records use. A triage of another format,
missing one of those keys, or with an entry of another shape (a `node_id` has
to be `file.py::name`, a `module` a `.py` file or a folder ending in a slash)
exits 2, so a change in `regressions` output cannot pass as a run with nothing
to do. A valid triage with no test in it exits 0 with an empty result, and
`--tests` given JSON exits 2. `drift_check` is not read: the workflow decides
what to bisect, and writes a triage holding only that.

For a target the step follows the contract in `nightly_canary.py`: it checks
the path exists, since one absent from the commit is good. From `pytest
--collect-only -q <target>`, exit 2 (a module's import error) and 4 (a
`conftest.py`'s, which pytest loads at startup) are bad, 0 and 5 are good, and
any other status is a skip, as any status but 0 or 1 is when a node id runs.

Caps: `--budget-seconds` (1200) bounds the run, `--step-timeout` (300) the
whole of each step, sync and every collection included, and at most 50 tests
are bisected; the others get a range event. Every loop stops at the deadline,
and no child starts past it. The margin: a child running at its bound gets
SIGTERM and 2 s before SIGKILL, and `run` ends a `git bisect run` or a check
whose step ignored `--deadline` 15 s past its own bound. Git calls have their
own 120 s timeout, outside that margin. Every child that can run long has its own process group and writes
its output to a temporary file. A timeout sends the group SIGTERM, then
SIGKILL, so a killed `uv` leaves no pytest behind and a process that leaves
the group cannot hold a pipe open. A step that receives SIGTERM kills its
own children first. Process groups make this script POSIX only; the canary
runs it on Linux and macOS.

SIGTERM or SIGINT to `run` ends the running child's group, resets the bisect,
saves the partial result with the rest of the tests under the reason `stopped`,
and exits 128 plus the signal. A `run` killed outright leaves `git bisect run`
and its step alive, and those would write into the next run's bisect. So each
`run` holds a lock in the repository's git directory and passes it to its
children, and while any of them lives the next `run` exits 2. They end at the
earlier run's deadline, when the step aborts. A child that calls `setsid`
leaves the group, and neither signal nor timeout reaches it.

Events. The result names one event per culprit or range, with the event
keys the canary uses: `pr-<N>` for a culprit that merged as PR N,
`commit-<sha12>` for a culprit with no PR, and `range-<good12>-<bad12>`
when no culprit was found. The result file is written with `complete` false
before any work, so a missing folder for `--out` exits 2 at once and an earlier
run's result cannot pass for this one's. It is replaced atomically after every
change, with `complete` false until the run ends, so a run that dies keeps
what it found. Each event lists its tests, and a range event maps
each test to the reason it has no culprit: `budget`, `inconclusive`,
`unverified`, `rounds`, `cap`, `passes at bad`, `unmapped`, `renamed`,
`returned`, `nonlinear` or `stopped`, for a run that raised or got a signal.
Every input test ends in exactly one event. Each round keeps its `git bisect
log`.

Standard library only. The functions above `Git` are pure; the tests in
`tests/test_nightly_bisect_1786.py` run them on values, and run the
subcommands on temporary git repositories with a fake step, a fake `uv`
and a fake `gh`.

Exit status (`merge-points`, `run`): 0, or 1 for `run` when any test got
a range event, or 2 for a usage error, an unreadable input or a git
command that failed.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, cast
from xml.etree import ElementTree

RESULT_FORMAT = 1
# The `format` this script reads from `nightly_canary.py regressions`.
TRIAGE_FORMAT = 1

# `git bisect run`'s protocol.
STEP_GOOD = 0
STEP_BAD = 1
STEP_SKIP = 125
STEP_ABORT = 255

DEFAULT_BUDGET_SECONDS = 1200
DEFAULT_STEP_TIMEOUT = 300
MAX_TESTS = 50
MAX_ROUNDS = 3

# pytest's status when it collected no test.
_PYTEST_NO_TESTS = 5
# Its `--collect-only` statuses for a target: a module's import error is 2 and
# a `conftest.py`'s is 4.
_TARGET_BAD = (2, 4)
_TARGET_GOOD = (0, _PYTEST_NO_TESTS)
_GIT_TIMEOUT_SECONDS = 120
_API_TIMEOUT_SECONDS = 60
_PR_LIST_LIMIT = 500
# Revisions per `git bisect skip`, to keep each command line short.
_SKIP_CHUNK = 200
# Seconds `run` gives a step that ignores `--deadline` or `--step-timeout`.
_RUN_MARGIN_SECONDS = 15
# Seconds a process group gets between SIGTERM and SIGKILL.
_TERM_GRACE_SECONDS = 2
# Lines of `git bisect run` output a round keeps.
_OUTPUT_TAIL_LINES = 40
_WORKTREE_MARKER = ".nightly-bisect-worktree"
_MAX_SECONDS = 7 * 24 * 3600

REASON_BUDGET = "budget"
REASON_INCONCLUSIVE = "inconclusive"
REASON_UNVERIFIED = "unverified"
REASON_ROUNDS = "rounds"
REASON_CAP = "cap"
REASON_PASSES_AT_BAD = "passes at bad"
REASON_UNMAPPED = "unmapped"
REASON_RENAMED = "renamed"
REASON_RETURNED = "returned"
REASON_NONLINEAR = "nonlinear"
REASON_STOPPED = "stopped"


# ------------------------------------------------------------------ step

def step_verdict(
    *,
    past_deadline: bool,
    sync_rc: int | None = 0,
    collect_rc: int | None = 0,
    present: Sequence[str] = (),
    run_rc: int | None = None,
    module_rcs: Sequence[int | None] = (),
    unmapped: bool = False,
) -> int:
    """The step's exit status. A status of None is a timeout.

    `collect_rc` is pytest's status for `--collect-only` over the files of
    the node ids, or 0 when none of them exists; `present` holds the ids
    it collected. `module_rcs` holds the `--collect-only` status of each
    target that exists: a bad status there makes the commit bad whatever
    the other tests do. `unmapped` says an id names no
    file, so the commit cannot be judged.
    """
    if past_deadline:
        return STEP_ABORT
    if unmapped or sync_rc != 0:
        return STEP_SKIP
    if any(rc not in (*_TARGET_BAD, *_TARGET_GOOD) for rc in module_rcs):
        return STEP_SKIP
    if any(rc in _TARGET_BAD for rc in module_rcs):
        return STEP_BAD
    if collect_rc not in (0, _PYTEST_NO_TESTS):
        return STEP_SKIP
    if not present:
        return STEP_GOOD
    if run_rc == 0:
        return STEP_GOOD
    if run_rc == 1:
        return STEP_BAD
    return STEP_SKIP


def module_file(test_id: str) -> str | None:
    """The path a collection error names, or None for a node id.

    pytest reports a module that fails to import as a JUnit case with an
    empty classname and the module's dotted path as its name, so the id
    arrives as `tests.test_x` or `::tests.test_x`; a bare `tests/test_x.py`
    or `tests/sub/` is read the same way. Any id that names a function or
    class is a node id.
    """
    rest = test_id[2:] if test_id.startswith("::") else test_id
    if not rest or "::" in rest or any(c.isspace() for c in rest):
        return None
    if rest.endswith((".py", "/")):
        return rest if _is_relative(rest) else None
    parts = rest.split(".")
    if "/" in rest or "\\" in rest or not all(parts):
        return None
    return "/".join(parts) + ".py"


def _is_relative(path: str) -> bool:
    return not path.startswith("/") and ".." not in path.split("/")


def canonical_id(test_id: str, is_file: Callable[[str], bool]) -> str | None:
    """`test_id` as a node id, or as a target; None when it names no file.

    A JUnit id's classname joins the module path to the class names with
    dots. The one prefix that `is_file` accepts is the module; none, or more
    than one, names no file. `nightly_canary.py`'s `node_id` reads an id the
    same way, so the two scripts agree on what is mapped. `is_file` also
    answers for a folder, named with a trailing slash: a dotted target with
    no such module names that folder, as `collection_module` reads it there.
    """
    head, sep, rest = test_id.partition("::")
    if not sep or not head:
        path = module_file(test_id)
        if path is not None and path.endswith(".py") and not is_file(path) and is_file(f"{path[:-3]}/"):
            return f"{path[:-3]}/"
        return path
    if head.endswith(".py"):
        return test_id if _is_relative(head) else None
    parts = head.split(".")
    if "/" in head or "[" in head or "]" in head or not rest or not all(parts):
        return None
    cuts = [cut for cut in range(1, len(parts) + 1) if is_file("/".join(parts[:cut]) + ".py")]
    if len(cuts) != 1 or not all(part.isidentifier() for part in parts[cuts[0]:]):
        return None
    return "::".join(["/".join(parts[:cuts[0]]) + ".py", *parts[cuts[0]:], rest])


def test_files(ids: Iterable[str]) -> list[str]:
    """The file part of each node id, once each, in order; module ids are left out."""
    out: list[str] = []
    for node_id in ids:
        if module_file(node_id) is not None:
            continue
        path = node_id.split("::", 1)[0]
        if path not in out:
            out.append(path)
    return out


def collected_ids(stdout: str) -> set[str]:
    """The node ids `pytest --collect-only -q` printed, one per line."""
    return {line.strip() for line in stdout.splitlines() if "::" in line}


def present_ids(candidates: Iterable[str], collected: set[str]) -> list[str]:
    return [c for c in candidates if c in collected]


def junit_key(node_id: str) -> str:
    """The `classname::name` a JUnit report gives a node id.

    `tests/a/test_b.py::TestC::test_d[x]` is `tests.a.test_b.TestC::test_d[x]`.
    A `::` inside the parameter brackets stays in the name.
    """
    path, _, rest = node_id.partition("::")
    bracket = rest.find("[")
    head, params = (rest, "") if bracket < 0 else (rest[:bracket], rest[bracket:])
    parts = head.split("::")
    module = path[:-3] if path.endswith(".py") else path
    classname = ".".join([module.replace("/", "."), *parts[:-1]])
    return f"{classname}::{parts[-1]}{params}"


def failing_ids(junit_text: str, ids: Iterable[str]) -> list[str]:
    """The ids, in order, that the report shows as failed or errored."""
    root = ElementTree.fromstring(junit_text)
    red: set[str] = set()
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            red.add(f"{case.get('classname') or ''}::{case.get('name') or ''}")
    return [i for i in ids if junit_key(i) in red]


# ----------------------------------------------------------------- range

def select_points(order: Sequence[str], prs: Iterable[dict[str, Any]], bad: str) -> dict[str, int | None]:
    """Commit to PR number for each commit of `order` that a PR merged,
    plus `bad`, in history order. Empty when no PR merged one of them: the
    list says nothing about the range, so `bad` alone would be a false claim.
    Raises ValueError for a PR with no merge commit: a point could be missing,
    and a missing point shifts the blame to the next one."""
    by_oid: dict[str, int] = {}
    for pr in prs:
        merge = pr.get("mergeCommit")
        oid = cast(dict[str, Any], merge).get("oid") if isinstance(merge, dict) else None
        if not oid:
            raise ValueError(f"PR {pr.get('number')} has no merge commit in gh pr list")
        by_oid[str(oid)] = int(pr["number"])
    if not any(sha in by_oid for sha in order):
        return {}
    return {sha: by_oid.get(sha) for sha in order if sha in by_oid or sha == bad}


def usable_points(points: dict[str, int | None], order: Sequence[str]) -> dict[str, int | None]:
    """`points`, or `{}` when none of them names a PR on `order`."""
    return points if any(points.get(sha) is not None for sha in order) else {}


def merged_since(committed_at: int) -> str:
    """The search date for `gh pr list`: the UTC day before the commit,
    so a committer's time zone cannot move a merge out of the window."""
    day = datetime.fromtimestamp(committed_at, tz=timezone.utc) - timedelta(days=1)
    return day.strftime("%Y-%m-%d")


def pr_list_argv(since: str, repo: str | None) -> list[str]:
    argv = [
        "gh", "pr", "list", "--state", "merged",
        "--search", f"merged:>={since}",
        "--json", "number,mergeCommit,title,url",
        "--limit", str(_PR_LIST_LIMIT),
    ]
    if repo:
        argv += ["--repo", repo]
    return argv


def skip_list(in_range: Sequence[str], points: set[str] | None) -> list[str]:
    """Commits to `git bisect skip`: none without merge-point data."""
    if points is None:
        return []
    return [c for c in in_range if c not in points]


def preceding_point(order: Sequence[str], points: set[str] | None, sha: str, lo: str) -> str:
    """The merge point before `sha` on `order`, or `lo` when none is.
    `order` runs oldest first and leaves `lo` out."""
    for commit in reversed(order[: order.index(sha)]):
        if points is None or commit in points:
            return commit
    return lo


def following_point(order: Sequence[str], points: set[str] | None, sha: str) -> str | None:
    """The merge point after `sha` on `order`, or None when none is."""
    for commit in order[order.index(sha) + 1 :]:
        if points is None or commit in points:
            return commit
    return None


def points_between(order: Sequence[str], points: set[str] | None, bad: str) -> list[str]:
    """Merge points the bisect has to test: those on `order` before `bad`."""
    return [c for c in order[: order.index(bad)] if points is None or c in points]


@dataclass(frozen=True)
class Outcome:
    culprit: str | None
    good: str
    bad: str


def read_outcome(
    order: Sequence[str], points: set[str] | None, lo: str, bad_ref: str, good_refs: Iterable[str],
) -> Outcome:
    """What the bisect refs say. The bad commit is the culprit when the
    merge point before it is good; otherwise the range runs from the
    newest good commit to the bad one."""
    position = {sha: i for i, sha in enumerate(order)}
    goods = {g for g in good_refs if g in position and position[g] < position[bad_ref]}
    newest_good = max(goods, key=position.__getitem__) if goods else lo
    before = preceding_point(order, points, bad_ref, lo)
    if before == lo or before in goods:
        return Outcome(culprit=bad_ref, good=before, bad=bad_ref)
    return Outcome(culprit=None, good=newest_good, bad=bad_ref)


def event_for_culprit(sha: str, pr: int | None, tests: list[str], preceding: str) -> dict[str, Any]:
    if pr is not None:
        return {"key": f"pr-{pr}", "kind": "pr", "pr": pr, "sha": sha, "preceding": preceding, "tests": tests}
    return {"key": f"commit-{sha[:12]}", "kind": "commit", "sha": sha, "preceding": preceding, "tests": tests}


def event_for_range(good: str, bad: str, tests: list[str], reason: str) -> dict[str, Any]:
    return {
        "key": f"range-{good[:12]}-{bad[:12]}", "kind": "range",
        "good": good, "bad": bad, "reasons": {test: reason for test in tests}, "tests": tests,
    }


def add_event(events: list[dict[str, Any]], event: dict[str, Any]) -> None:
    """Append `event`, or merge its tests and reasons into the event with its key."""
    for existing in events:
        if existing["key"] == event["key"]:
            existing["tests"] = [*existing["tests"], *(t for t in event["tests"] if t not in existing["tests"])]
            if "reasons" in event:
                existing["reasons"] = {**event["reasons"], **existing["reasons"]}
            return
    events.append(event)


def parse_tests(text: str) -> list[str]:
    """One node id per line; blank lines and `#` comments are ignored."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and line not in out:
            out.append(line)
    return out


def parse_points(data: Any) -> dict[str, int | None]:
    if not isinstance(data, dict):
        raise ValueError("the points file must hold a JSON object")
    out: dict[str, int | None] = {}
    for sha, pr in cast(dict[str, Any], data).items():
        if pr is not None and (not isinstance(pr, int) or isinstance(pr, bool)):
            raise ValueError(f"the points file maps {sha} to {pr!r}, not a PR number or null")
        out[sha] = pr
    return out


# ------------------------------------------------------------------- git

# What `git rev-parse --local-env-vars` lists: the variables that tie a
# process to one repository, plus the prefix `git bisect run` exports.
_GIT_REPO_VARS = frozenset({
    "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_CONFIG", "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS", "GIT_DIR", "GIT_GRAFT_FILE", "GIT_IMPLICIT_WORK_TREE", "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS", "GIT_OBJECT_DIRECTORY", "GIT_PREFIX", "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE", "GIT_WORK_TREE",
})


def repo_free_env() -> dict[str, str]:
    """This process's environment without git's repository variables."""
    return {k: v for k, v in os.environ.items() if k not in _GIT_REPO_VARS}


class Git:
    """`git` in one directory, with a timeout on every call."""

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def run(self, *args: str, check: bool = True, timeout: float = _GIT_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
        r = subprocess.run(
            ["git", *args], cwd=self.cwd, capture_output=True, text=True, env=repo_free_env(),
            encoding="utf-8", errors="replace", check=False, timeout=timeout,
        )
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return r

    def out(self, *args: str) -> str:
        return self.run(*args).stdout.strip()

    def commit(self, rev: str) -> str:
        return self.out("rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")

    def lines(self, *args: str) -> list[str]:
        return [line for line in self.out(*args).splitlines() if line]


def ancestry(git: Git, good: str, bad: str) -> list[str]:
    return git.lines("rev-list", "--reverse", "--ancestry-path", f"{good}..{bad}")


def is_linear(git: Git, good: str, bad: str) -> bool:
    """True when each commit of `good..bad` has the one before it as its only parent."""
    previous = good
    for row in git.lines("rev-list", "--reverse", "--parents", f"{good}..{bad}"):
        sha, *parents = row.split()
        if parents != [previous]:
            return False
        previous = sha
    return True


def resolve_range(git: Git, good: str, bad: str) -> tuple[str, str]:
    """Both commits as full shas, with `good` a proper ancestor of `bad`."""
    good_sha, bad_sha = git.commit(good), git.commit(bad)
    if good_sha == bad_sha:
        raise ValueError(f"--good and --bad are the same commit {good_sha[:12]}")
    if git.run("merge-base", "--is-ancestor", good_sha, bad_sha, check=False).returncode != 0:
        raise ValueError(f"{good_sha[:12]} is not an ancestor of {bad_sha[:12]}")
    return good_sha, bad_sha


# -------------------------------------------------------- process groups

# The groups this process has running, for the step's SIGTERM handler.
_LIVE_GROUPS: set[int] = set()
# What every child inherits: `run`'s lock, so an orphan keeps it held.
_INHERITED_FDS: tuple[int, ...] = ()


class Interrupted(Exception):
    """`run` got SIGTERM or SIGINT."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signum}")
        self.signum = signum


def _interrupt(signum: int, _frame: object) -> None:
    """`run`'s signal handler. A second signal is ignored, so the cleanup finishes."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise Interrupted(signum)


def _signal_group(group: int, signum: int) -> None:
    try:
        os.killpg(group, signum)
    except OSError:
        pass  # The group has already ended.


def _end_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM the group and give all of it the grace to end: the step's handler
    has to run before the SIGKILL that follows, even once `git` has gone."""
    _signal_group(proc.pid, signal.SIGTERM)
    limit = time.monotonic() + _TERM_GRACE_SECONDS
    while time.monotonic() < limit:
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
        except OSError:
            return
        time.sleep(0.05)


def run_group(
    argv: list[str], timeout: float, *, cwd: Path | None = None, capture: bool = True,
) -> tuple[int | None, str]:
    """Run `argv` in its own process group and end the whole group after it.

    Status None is a timeout; a missing program is status 127. Output goes
    to a temporary file, not a pipe, so a process that escapes the group
    cannot keep this call waiting. With `capture` the output is stdout and
    stderr together.
    """
    with tempfile.TemporaryFile() as out:
        try:
            proc = subprocess.Popen(
                argv, cwd=cwd, env=repo_free_env(), stdin=subprocess.DEVNULL,
                stdout=out if capture else None, stderr=subprocess.STDOUT if capture else None,
                start_new_session=True, pass_fds=_INHERITED_FDS,
            )
        except FileNotFoundError:
            return 127, ""
        _LIVE_GROUPS.add(proc.pid)
        rc: int | None = None
        try:
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _end_group(proc)
            except Interrupted:
                _end_group(proc)
                raise
        finally:
            _signal_group(proc.pid, signal.SIGKILL)
            _LIVE_GROUPS.discard(proc.pid)
        proc.wait(timeout=_TERM_GRACE_SECONDS)
        out.seek(0)
        return rc, out.read().decode("utf-8", "replace")


def _die_with_children(_signum: int, _frame: object) -> None:
    """The step's SIGTERM handler: its children have their own groups, so a
    signal to the step's group does not reach them."""
    for group in list(_LIVE_GROUPS):
        _signal_group(group, signal.SIGKILL)
    os._exit(STEP_ABORT)


# ------------------------------------------------------------- the step

def _bounded(argv: list[str], end: float, capture: bool = False) -> tuple[int | None, str]:
    """Run `argv` until `end` (epoch seconds), by `run_group`. Past `end` it
    starts nothing, and the status is the None of a timeout."""
    left = end - time.time()
    if left <= 0:
        return None, ""
    rc, output = run_group(argv, left, capture=capture)
    return rc, output if capture else ""


def _on_disk(path: str) -> bool:
    """`canonical_id`'s `is_file` for the checked-out tree."""
    return Path(path).is_dir() if path.endswith("/") else Path(path).is_file()


def _sync_argv(python: str | None, extras: Sequence[str]) -> list[str]:
    argv = ["uv", "sync", "--frozen"]
    for extra in extras:
        argv += ["--extra", extra]
    if python:
        argv += ["--python", python]
    return argv


_PYTEST = ["uv", "run", "--no-sync", "pytest", "-p", "no:cacheprovider"]


def cmd_step(args: argparse.Namespace) -> int:
    given = parse_tests(args.tests.read_text(encoding="utf-8"))
    sync = _sync_argv(args.python, args.extra)
    mapped = {test: canonical_id(test, _on_disk) for test in given}
    ids = [test for test in mapped.values() if test is not None]
    as_given = {canon: test for test, canon in mapped.items() if canon is not None}
    if args.dry_run:
        print(f"dry run: would run {shlex.join(sync)}")
        print(f"dry run: would collect {len(ids)} test id(s) from {len(test_files(ids))} file(s) and run those present")
        return 0
    now = time.time()
    if now >= args.deadline:
        print("nightly_bisect step: past the deadline; aborting the bisect", file=sys.stderr)
        return step_verdict(past_deadline=True)
    if None in mapped.values():
        unread = [test for test, canon in mapped.items() if canon is None]
        print(f"nightly_bisect step: no file for {unread}; skipping this commit", file=sys.stderr)
        return step_verdict(past_deadline=False, unmapped=True)
    end = min(args.deadline, now + args.step_timeout)
    failed: list[str] = []
    absent: list[str] = []
    previous = signal.signal(signal.SIGTERM, _die_with_children)
    try:
        sync_rc, _ = _bounded(sync, end)
        if sync_rc != 0:
            return step_verdict(past_deadline=False, sync_rc=sync_rc)
        # A target is judged by whether it collects: absent or collecting
        # is good, a collection error is bad.
        module_rcs: list[int | None] = []
        for test_id in ids:
            path = module_file(test_id)
            if path is None:
                continue
            if not _on_disk(path):
                absent.append(test_id)
                continue
            rc, _ = _bounded([*_PYTEST, "--collect-only", "-q", path], end, capture=True)
            module_rcs.append(rc)
            if rc in _TARGET_BAD:
                failed.append(test_id)
        node_ids = [i for i in ids if module_file(i) is None]
        files = [f for f in test_files(ids) if Path(f).is_file()]
        absent += [i for i in node_ids if i.split("::", 1)[0] not in files]
        if not files:
            return step_verdict(past_deadline=False, module_rcs=module_rcs)
        collect_rc, listing = _bounded([*_PYTEST, "--collect-only", "-q", *files], end, capture=True)
        present = present_ids(ids, collected_ids(listing))
        if collect_rc in (0, _PYTEST_NO_TESTS):
            absent += [i for i in node_ids if i not in absent and i not in present]
        if collect_rc not in (0, _PYTEST_NO_TESTS) or not present:
            return step_verdict(past_deadline=False, collect_rc=collect_rc, present=present, module_rcs=module_rcs)
        with tempfile.TemporaryDirectory() as tmp:
            junit = Path(tmp) / "step.xml"
            run_rc, _ = _bounded([*_PYTEST, "-q", f"--junitxml={junit}", *present], end)
            if junit.is_file():
                try:
                    failed += failing_ids(junit.read_text(encoding="utf-8"), present)
                except ElementTree.ParseError:
                    pass
        return step_verdict(
            past_deadline=False, collect_rc=collect_rc, present=present, run_rc=run_rc, module_rcs=module_rcs,
        )
    finally:
        if args.failed_out is not None:
            args.failed_out.write_text("".join(f"{as_given[t]}\n" for t in failed), encoding="utf-8")
        if args.absent_out is not None:
            args.absent_out.write_text("".join(f"{as_given[t]}\n" for t in absent), encoding="utf-8")
        signal.signal(signal.SIGTERM, previous)


# -------------------------------------------------------------- the run

def default_step_cmd() -> list[str]:
    """This script's `step`, by absolute path: `git bisect run` runs it in
    the worktree, where a relative path names the bisected commit's copy."""
    return [sys.executable, str(Path(__file__).resolve()), "step"]


@dataclass(frozen=True)
class Check:
    """One step run outside the bisect: its status and the tests it named."""
    rc: int
    failed: list[str]
    absent: list[str]

    @property
    def judged(self) -> bool:
        """False for a status with no verdict, and for a 1 that names no test,
        which is what a missing or corrupt JUnit report leaves."""
        return self.rc == STEP_GOOD or (self.rc == STEP_BAD and bool(self.failed))


@dataclass
class Bisector:
    git: Git
    worktree: Path
    bad: str
    points: dict[str, int | None]
    step_cmd: list[str]
    step_args: list[str]
    deadline: float
    step_timeout: int
    scratch: Path
    events: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    rounds: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    reset_errors: list[str] = field(default_factory=list[str])
    on_change: Callable[[], None] = lambda: None

    @property
    def point_set(self) -> set[str] | None:
        """The merge points, `bad` included, or None without data."""
        return set(self.points) | {self.bad} if self.points else None

    def _wt(self) -> Git:
        return Git(self.worktree)

    def _step(self, tests_file: Path, failed_out: Path | None = None, absent_out: Path | None = None) -> list[str]:
        argv = [
            *self.step_cmd, "--tests", str(tests_file),
            "--deadline", f"{self.deadline:.3f}", "--step-timeout", str(self.step_timeout),
            *self.step_args,
        ]
        if failed_out is not None:
            argv += ["--failed-out", str(failed_out)]
        if absent_out is not None:
            argv += ["--absent-out", str(absent_out)]
        return argv

    def _write_tests(self, name: str, tests: list[str]) -> Path:
        path = self.scratch / name
        path.write_text("".join(f"{t}\n" for t in tests), encoding="utf-8")
        return path

    def _check(self, sha: str, tests: list[str]) -> Check:
        """The step at `sha` over `tests`, run outside a bisect. Past the
        deadline it reads as the step's abort, and nothing runs."""
        left = self.deadline - time.time()
        if left <= 0:
            return Check(STEP_ABORT, [], [])
        tests_file = self._write_tests("check.txt", tests)
        failed_out = self.scratch / "check-failed.txt"
        absent_out = self.scratch / "check-absent.txt"
        for path in (failed_out, absent_out):
            path.write_text("", encoding="utf-8")
        self._wt().run("checkout", "-q", "--force", "--detach", sha)
        rc, _ = run_group(
            self._step(tests_file, failed_out, absent_out),
            min(self.step_timeout, left) + _RUN_MARGIN_SECONDS, cwd=self.worktree,
        )
        failed = [t for t in parse_tests(failed_out.read_text(encoding="utf-8")) if t in tests]
        absent = [t for t in parse_tests(absent_out.read_text(encoding="utf-8")) if t in tests]
        return Check(STEP_SKIP if rc is None else rc, failed, absent)

    def bisect_round(self, lo: str, tests: list[str], n: int) -> tuple[Outcome, dict[str, Any]]:
        """One `git bisect` from `lo` to `bad` over `tests`."""
        order = ancestry(self.git, lo, self.bad)
        points = self.point_set
        tests_file = self._write_tests(f"round-{n}.txt", tests)
        record: dict[str, Any] = {"good": lo, "bad": self.bad, "tests": tests}
        if not points_between(order, points, self.bad):
            # Nothing between the two to test: `bad` follows `lo` directly.
            record.update(log="", output="")
            return Outcome(culprit=self.bad, good=lo, bad=self.bad), record
        wt = self._wt()
        skips = skip_list(order, points)
        record["skipped"] = len(skips)
        try:
            wt.run("bisect", "start", self.bad, lo)
            for i in range(0, len(skips), _SKIP_CHUNK):
                if time.time() >= self.deadline:
                    break  # The step aborts at once, so the unskipped commits never run.
                wt.run("bisect", "skip", *skips[i : i + _SKIP_CHUNK])
            rc, output = run_group(
                ["git", "bisect", "run", *self._step(tests_file)],
                max(0.0, self.deadline - time.time()) + _RUN_MARGIN_SECONDS, cwd=self.worktree,
            )
            if rc is None:
                output += "\ngit bisect run timed out"
            bad_ref = wt.out("rev-parse", "--verify", "refs/bisect/bad")
            goods = [
                sha for sha, ref in (line.split(" ", 1) for line in wt.lines(
                    "for-each-ref", "--format=%(objectname) %(refname)", "refs/bisect/",
                ))
                if ref.startswith("refs/bisect/good-")
            ]
            record["log"] = wt.run("bisect", "log", check=False).stdout
            record["output"] = "\n".join(output.splitlines()[-_OUTPUT_TAIL_LINES:])
        finally:
            self._reset(wt)
        return read_outcome(order, points, lo, bad_ref, goods), record

    def _event(self, event: dict[str, Any]) -> None:
        add_event(self.events, event)
        self.on_change()

    def _reset(self, wt: Git) -> None:
        """End the bisect, on a clean tree when the first try fails."""
        try:
            r = wt.run("bisect", "reset", check=False)
            if r.returncode != 0:
                wt.run("reset", "-q", "--hard", check=False)
                r = wt.run("bisect", "reset", check=False)
            error = (r.stderr.strip() or f"exit status {r.returncode}") if r.returncode != 0 else ""
        except subprocess.TimeoutExpired as exc:
            error = str(exc)
        if error:
            self.reset_errors.append(error)
            print(f"::warning::git bisect reset failed: {error}", file=sys.stderr)

    def attribute(
        self, lo: str, outcome: Outcome, remaining: list[str], record: dict[str, Any],
    ) -> tuple[list[str], list[str]] | None:
        """The tests blamed on the culprit and those for the next round, or
        None when the round ended in range events."""
        culprit = cast(str, outcome.culprit)
        first = self._check(culprit, remaining)
        before = self._check(outcome.good, remaining)
        record.update(verify={"culprit_rc": first.rc, "preceding": outcome.good, "preceding_rc": before.rc})
        if STEP_ABORT in (first.rc, before.rc):
            self._event(event_for_range(outcome.good, outcome.bad, remaining, REASON_BUDGET))
            return None
        if first.rc != STEP_BAD or not first.judged or not before.judged:
            self._event(event_for_range(lo, self.bad, remaining, REASON_UNVERIFIED))
            return None
        earlier = [t for t in first.failed if t in before.failed]
        if earlier:
            self._event(event_for_range(lo, culprit, earlier, REASON_UNVERIFIED))
        undecided = [t for t in remaining if t not in earlier]
        broke = [t for t in first.failed if t not in earlier]
        blamed: list[str] = []
        if broke:
            again = self._check(culprit, broke)
            record["verify"]["again_rc"] = again.rc
            if again.rc == STEP_ABORT:
                self._event(event_for_range(outcome.good, outcome.bad, undecided, REASON_BUDGET))
                return None
            if not again.judged:
                self._event(event_for_range(lo, self.bad, undecided, REASON_UNVERIFIED))
                return None
            blamed = again.failed
        settled = [*earlier, *blamed]
        blamed = self._still_broken(lo, outcome, blamed, record)
        blamed = self._new_here(lo, outcome, blamed, before.absent)
        return blamed, [t for t in remaining if t not in settled]

    def _new_here(self, lo: str, outcome: Outcome, blamed: list[str], absent_before: list[str]) -> list[str]:
        """Those of `blamed` that the culprit did not bring back. The others
        existed before under another name or before a gap, and read good
        there, so their real break can lie anywhere up to the culprit."""
        culprit = cast(str, outcome.culprit)
        absent = [t for t in blamed if t in absent_before]
        if not absent:
            return blamed
        moved_files = self.git.lines(
            "diff", "--name-only", "--find-renames", "--find-copies", "--diff-filter=RC", outcome.good, culprit,
        )
        moved = [t for t in absent if t.split("::", 1)[0] in moved_files]
        if moved:
            self._event(event_for_range(lo, culprit, moved, REASON_RENAMED))
        returned: list[str] = []
        unexplained = [t for t in absent if t not in moved]
        if unexplained and outcome.good != lo:
            origin = self._check(lo, unexplained)
            if origin.rc not in (STEP_GOOD, STEP_BAD):
                # The round's first commit could not say the test is new.
                reason = REASON_BUDGET if origin.rc == STEP_ABORT else REASON_UNVERIFIED
                self._event(event_for_range(lo, culprit, unexplained, reason))
                return [t for t in blamed if t not in absent]
            returned = [t for t in unexplained if t not in origin.absent]
        if returned:
            self._event(event_for_range(lo, culprit, returned, REASON_RETURNED))
        return [t for t in blamed if t not in moved and t not in returned]

    def _still_broken(self, lo: str, outcome: Outcome, blamed: list[str], record: dict[str, Any]) -> list[str]:
        """Those of `blamed` that do not pass at the merge point after the
        culprit. The others get a range event: a false bad twice at the
        culprit looks like a break until a later commit passes."""
        culprit = cast(str, outcome.culprit)
        after = following_point(ancestry(self.git, lo, self.bad), self.point_set, culprit)
        if not blamed or after is None or after == self.bad:
            return blamed
        later = self._check(after, blamed)
        record["verify"].update(following=after, following_rc=later.rc)
        if later.rc == STEP_ABORT:
            self._event(event_for_range(outcome.good, outcome.bad, blamed, REASON_BUDGET))
            return []
        if later.rc not in (STEP_GOOD, STEP_BAD):
            return blamed  # This commit cannot be judged, which says nothing about the culprit.
        red = (later.failed or blamed) if later.rc == STEP_BAD else []
        passed = [t for t in blamed if t not in red]
        if passed:
            self._event(event_for_range(lo, self.bad, passed, REASON_UNVERIFIED))
        return [t for t in blamed if t in red]

    def settle(self, good: str, tests: list[str]) -> None:
        """Give any of `tests` that no event names a range: the run stopped early."""
        named = {t for event in self.events for t in event["tests"]}
        rest = [t for t in tests if t not in named]
        if rest:
            add_event(self.events, event_for_range(good, self.bad, rest, REASON_STOPPED))

    def bisect(self, good: str, tests: list[str]) -> None:
        lo, remaining = good, list(tests)
        for n in range(1, MAX_ROUNDS + 1):
            if not remaining:
                return
            if lo == self.bad:
                self._event(event_for_range(lo, self.bad, remaining, REASON_PASSES_AT_BAD))
                return
            if time.time() >= self.deadline:
                self._event(event_for_range(lo, self.bad, remaining, REASON_BUDGET))
                return
            outcome, record = self.bisect_round(lo, remaining, n)
            self.rounds.append(record)
            if outcome.culprit is None:
                record["culprit"] = None
                self._event(event_for_range(outcome.good, outcome.bad, remaining, REASON_INCONCLUSIVE))
                return
            record["culprit"] = outcome.culprit
            attributed = self.attribute(lo, outcome, remaining, record)
            if attributed is None:
                return
            blamed, remaining = attributed
            if blamed:
                self._event(event_for_culprit(outcome.culprit, self.points.get(outcome.culprit), blamed, outcome.good))
            lo = outcome.culprit
        if remaining:
            self._event(event_for_range(lo, self.bad, remaining, REASON_ROUNDS))


def tree_paths(files: Iterable[str]) -> set[str]:
    """`files`, and each folder above one with a trailing slash."""
    return {*files, *(f"{parent}/" for name in files for parent in PurePosixPath(name).parents if parent.name)}


def map_ids(tests: Iterable[str], files: set[str]) -> tuple[dict[str, list[str]], list[str]]:
    """Each id that names a path in `files` under its canonical id, with the
    ids given for it, and the ids that name none."""
    mapped: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for test in tests:
        canon = canonical_id(test, files.__contains__)
        if canon is None or canon.split("::", 1)[0] not in files:
            unmapped.append(test)
        else:
            mapped.setdefault(canon, []).append(test)
    return mapped, unmapped


def map_triage(triage: Any, files: set[str]) -> tuple[dict[str, list[str]], list[str]]:
    """As `map_ids`, for `nightly_canary.py regressions` output: the mapping
    is already done, so each entry only has to name a file in `files`."""
    if not isinstance(triage, dict):
        raise ValueError("the triage file must hold a JSON object")
    data = cast(dict[str, Any], triage)
    if type(data.get("format")) is not int or data["format"] != TRIAGE_FORMAT:
        raise ValueError(f"the triage file has format {data.get('format')!r}, not {TRIAGE_FORMAT}")
    entries = [
        *_triage_rows(data, "candidates", "node_id", _is_node_id),
        *_triage_rows(data, "new_failing", "node_id", _is_node_id),
        *_triage_rows(data, "collection_errors", "module", lambda path: module_file(path) == path),
    ]
    mapped: dict[str, list[str]] = {}
    unmapped: list[str] = []
    seen: set[str] = set()
    for test, canon in entries:
        if test in seen:
            continue
        seen.add(test)
        if canon is None or canon.split("::", 1)[0] not in files:
            unmapped.append(test)
        else:
            mapped.setdefault(canon, []).append(test)
    return mapped, unmapped


def _is_node_id(text: str) -> bool:
    path, sep, name = text.partition("::")
    return bool(sep and name) and path.endswith(".py")


def _triage_rows(
    data: dict[str, Any], key: str, target: str, valid: Callable[[str], bool],
) -> list[tuple[str, str | None]]:
    """(test, target) of each row of `data[key]`; ValueError unless the shape is right."""
    rows = data[key]
    if not isinstance(rows, list):
        raise ValueError(f"the triage's {key} is not a list")
    out: list[tuple[str, str | None]] = []
    for row in cast(list[dict[str, Any]], rows):
        test, path = row["test"], row[target]
        if not isinstance(test, str) or not test or not (path is None or (isinstance(path, str) and valid(path))):
            raise ValueError(f"the triage's {key} holds {row!r}")
        out.append((test, path))
    return out


def worktree_refusal(path: Path, default: Path) -> str | None:
    """Why `run` may not use `path` as its worktree, or None. A folder this
    script does not own may hold someone's work, so it is never removed."""
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        return f"{path} is not a folder"
    if path == default or (path / _WORKTREE_MARKER).is_file():
        return None
    if path.exists() and any(path.iterdir()):
        return f"{path} holds files and this script did not create a worktree there"
    return None


def _prepare_worktree(git: Git, path: Path, default: Path, good: str) -> None:
    """A detached worktree at `good`, replacing what a previous run left: a
    worktree, a registered folder missing its `.git` file, or a registration
    whose folder is gone. `worktree_refusal` has passed `path`."""
    if path.exists() and any(path.iterdir()):
        # A test can leave a read-only folder, which would stop the removal.
        subprocess.run(["chmod", "-R", "u+rwx", str(path)], capture_output=True, check=False,
                       timeout=_GIT_TIMEOUT_SECONDS)
        git.run("worktree", "remove", "--force", "--force", str(path), check=False)
        if path.exists():
            shutil.rmtree(path)
    git.run("worktree", "prune")
    git.run("worktree", "add", "--detach", str(path), good)
    if path != default:
        # The default needs no marker, and an untracked file there could
        # fail a test that reads `git status`.
        (path / _WORKTREE_MARKER).write_text(f"{git.cwd}\n", encoding="utf-8")


@contextlib.contextmanager
def _run_lock(git: Git) -> Iterator[None]:
    """Hold the repository's lock for the block, or raise ValueError when a
    run, or what a killed run left alive, holds it. Children inherit it."""
    global _INHERITED_FDS
    path = git.cwd / git.out("rev-parse", "--git-common-dir") / "nightly-bisect.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(f"{path} is held: an earlier run, or what it left alive, is still running") from exc
        _INHERITED_FDS = (fd,)
        yield
    finally:
        _INHERITED_FDS = ()
        os.close(fd)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    """Replace `path` in one step, from a temporary file beside it."""
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.resolve().parent, prefix=f".{path.name}.", delete=False,
    ) as tmp:
        tmp.write(json.dumps(data, indent=1) + "\n")
    try:
        os.replace(tmp.name, path)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def cmd_merge_points(args: argparse.Namespace) -> int:
    git = Git(args.repo_root.resolve())
    good, bad = resolve_range(git, args.good, args.bad)
    order = ancestry(git, good, bad)
    since = merged_since(int(git.out("show", "-s", "--format=%ct", good)))
    argv = pr_list_argv(since, args.repo)
    if args.dry_run:
        print(f"dry run: {len(order)} commit(s) from {good[:12]} to {bad[:12]}")
        print(f"dry run: would run {shlex.join(argv)}")
        print(f"dry run: would write {args.out}")
        return 0
    try:
        r = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False, timeout=_API_TIMEOUT_SECONDS,
        )
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip() or f"exit status {r.returncode}")
        prs = json.loads(r.stdout)
        if not isinstance(prs, list):
            raise ValueError("gh pr list did not return a list")
        rows = cast(list[dict[str, Any]], prs)
        if len(rows) >= _PR_LIST_LIMIT:
            raise ValueError(f"gh pr list returned its limit of {_PR_LIST_LIMIT}, so it may be cut short")
        points = select_points(order, rows, bad)
        if not points:
            raise ValueError("gh pr list names no merge in the range")
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError,
            subprocess.TimeoutExpired) as exc:
        print(f"::warning::merge points unavailable, so the bisect tests every commit: {exc}", file=sys.stderr)
        points = {}
    _write_json(args.out, points)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = args.repo_root.resolve()
    git = Git(root)
    good, bad = resolve_range(git, args.good, args.bad)
    files = tree_paths(git.lines("ls-tree", "-r", "--name-only", bad))
    if args.triage is not None:
        mapped, unmapped = map_triage(_read_json(args.triage), files)
    else:
        text = args.tests.read_text(encoding="utf-8")
        if text.lstrip().startswith("{"):
            raise ValueError(f"{args.tests} holds JSON; a triage goes to --triage")
        mapped, unmapped = map_ids(parse_tests(text), files)
    tests = [*(given for ids in mapped.values() for given in ids), *unmapped]
    if not tests and args.triage is None:
        raise ValueError(f"{args.tests} names no test")
    given_points = parse_points(_read_json(args.points)) if args.points is not None else {}
    points = usable_points(given_points, ancestry(git, good, bad))
    if given_points and not points:
        print("::warning::the points file names no PR in the range, so the bisect tests every commit", file=sys.stderr)
    canonical = list(mapped)
    bisected, over_cap = canonical[:MAX_TESTS], canonical[MAX_TESTS:]
    step_cmd = shlex.split(args.step_cmd) if args.step_cmd else default_step_cmd()
    step_args = _step_passthrough(args)
    # Git runs in `root`, so a relative path would land under it.
    default_worktree = root / ".bisect"
    worktree = args.worktree.resolve() if args.worktree is not None else default_worktree
    refusal = worktree_refusal(worktree, default_worktree)
    if refusal is not None:
        raise ValueError(refusal)
    linear = is_linear(git, good, bad)
    if args.dry_run:
        in_range = ancestry(git, good, bad)
        point_set = set(points) | {bad} if points else None
        if not linear:
            print("dry run: the range is not one chain of commits, so it would test nothing")
        print(f"dry run: {len(in_range)} commit(s) from {good[:12]} to {bad[:12]}, "
              f"{len(in_range) - len(skip_list(in_range, point_set))} to test, {len(bisected)} test(s)")
        if unmapped:
            print(f"dry run: {len(unmapped)} id(s) name no file at {bad[:12]}: {unmapped}")
        print(f"dry run: would add the worktree {worktree} and run {shlex.join([*step_cmd, *step_args])}")
        print(f"dry run: would write {args.out}")
        return 0
    deadline = time.time() + args.budget_seconds

    def save(complete: bool) -> None:
        _write_json(args.out, {
            "format": RESULT_FORMAT, "complete": complete, "good": good, "bad": bad, "tests": tests,
            "merge_points": len(points), "events": [_as_given(e, mapped) for e in bisector.events],
            "rounds": bisector.rounds, "reset_errors": bisector.reset_errors,
        })

    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGTERM, signal.SIGINT)}
    with _run_lock(git), tempfile.TemporaryDirectory() as scratch:
        bisector = Bisector(
            git=git, worktree=worktree, bad=bad, points=points, step_cmd=step_cmd,
            step_args=step_args, deadline=deadline, step_timeout=args.step_timeout, scratch=Path(scratch),
            on_change=lambda: save(False),
        )
        complete = False
        try:
            for signum in previous:
                signal.signal(signum, _interrupt)
            save(False)
            if bisected and not linear:
                add_event(bisector.events, event_for_range(good, bad, bisected, REASON_NONLINEAR))
            elif bisected:
                _prepare_worktree(git, worktree, default_worktree, good)
                bisector.bisect(good, bisected)
            complete = True
        except Interrupted as exc:
            print(f"nightly_bisect: stopped by {exc}", file=sys.stderr)
            return 128 + exc.signum
        finally:
            bisector.settle(good, bisected)
            if over_cap:
                add_event(bisector.events, event_for_range(good, bad, over_cap, REASON_CAP))
            if unmapped:
                add_event(bisector.events, event_for_range(good, bad, unmapped, REASON_UNMAPPED))
            save(complete)
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    for event in bisector.events:
        print(f"{event['key']}: {len(event['tests'])} test(s)")
    return 1 if any(e["kind"] == "range" for e in bisector.events) else 0


def _as_given(event: dict[str, Any], mapped: dict[str, list[str]]) -> dict[str, Any]:
    """`event` with its tests named as `--tests` named them."""
    out = dict(event)
    out["tests"] = [given for test in event["tests"] for given in mapped.get(test, [test])]
    if "reasons" in event:
        out["reasons"] = {
            given: reason for test, reason in event["reasons"].items() for given in mapped.get(test, [test])
        }
    return out


def _step_passthrough(args: argparse.Namespace) -> list[str]:
    out: list[str] = []
    for extra in args.extra:
        out += ["--extra", extra]
    if args.python:
        out += ["--python", args.python]
    return out


def _seconds(text: str) -> int:
    """A whole number of seconds that adding to the clock cannot overflow."""
    value = int(text)
    if not 0 <= value <= _MAX_SECONDS:
        raise argparse.ArgumentTypeError(f"must be from 0 to {_MAX_SECONDS}")
    return value


def _epoch(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be a finite number")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="write no file, call no API, change no git state")
    env = argparse.ArgumentParser(add_help=False)
    env.add_argument("--python", default=None, help="the Python version uv syncs")
    env.add_argument("--extra", action="append", default=[], help="an extra uv syncs; repeatable")
    env.add_argument("--step-timeout", type=_seconds, default=DEFAULT_STEP_TIMEOUT)

    p = sub.add_parser("merge-points", parents=[common], help="list the commits PRs merged between two commits")
    p.add_argument("--good", required=True)
    p.add_argument("--bad", required=True)
    p.add_argument("--repo", default=None, help="OWNER/NAME for gh; the checkout's remote when omitted")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=cmd_merge_points)

    p = sub.add_parser("run", parents=[common, env], help="bisect the tests from --good to --bad")
    p.add_argument("--good", required=True)
    p.add_argument("--bad", required=True)
    tests = p.add_mutually_exclusive_group(required=True)
    tests.add_argument("--tests", type=Path, help="one node, JUnit or module id per line")
    tests.add_argument("--triage", type=Path, help="nightly_canary.py regressions output")
    p.add_argument("--points", type=Path, default=None, help="merge-points output; every commit is tested without it")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--worktree", type=Path, default=None, help="default: .bisect under --repo-root")
    p.add_argument("--budget-seconds", type=_seconds, default=DEFAULT_BUDGET_SECONDS)
    p.add_argument("--step-cmd", default=None, help="a command line that replaces this script's step")
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("step", parents=[common, env], help="judge the checked-out commit for git bisect run")
    p.add_argument("--tests", required=True, type=Path)
    p.add_argument("--deadline", required=True, type=_epoch, help="epoch seconds; past it the step exits 255")
    p.add_argument("--failed-out", type=Path, default=None, help="write the ids that failed here")
    p.add_argument("--absent-out", type=Path, default=None, help="write the ids that do not exist here")
    p.set_defaults(func=cmd_step)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    step = argv[:1] == ["step"]
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:
        if step and exc.code:
            return STEP_SKIP  # `git bisect run` reads a usage error's 2 as a bad commit.
        raise
    try:
        return int(args.func(args))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(f"nightly_bisect: {exc}", file=sys.stderr)
        # `git bisect run` reads 2 as a bad commit, so a step that cannot
        # judge its commit stops the bisect instead.
        return STEP_ABORT if step else 2
    except Exception as exc:
        # A traceback's status of 1 would read as a bad commit.
        print(f"nightly_bisect: {exc!r}", file=sys.stderr)
        return STEP_SKIP if step else 2


if __name__ == "__main__":
    sys.exit(main())
