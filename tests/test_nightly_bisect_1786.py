"""#1786: the nightly canary's bisect, `scripts/nightly_bisect.py`.

The pure functions run on values. The subcommands run on temporary git
repositories whose `state.txt` says, per commit, which tests fail.
A fake step stands in for uv and pytest through `--step-cmd` and logs the
commit each call ran on, so a test can name every commit the bisect
tested. One test runs the real step under a fake `uv` on PATH, and the
merge-point tests put a fake `gh` on PATH that logs its arguments. The
repositories live under a sandboxed HOME with `GIT_CONFIG_NOSYSTEM=1` and
signing off in their own config, so no outside git config reaches them.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

import nightly_bisect as nb  # noqa: E402

# Each bisect runs git and a Python step per commit it tests.
pytestmark = pytest.mark.timeout(120)

_A = "tests/test_x.py::test_a"
_B = "tests/test_x.py::test_b"
_C = "tests/test_x.py::test_c"
_D = "tests/test_x.py::test_d"

# The fake step: logs HEAD and its phase, `bisect` under `git bisect run`
# or `check` when given `--failed-out`. A test fails when `state.txt` marks
# it bad, and counts as absent when its file is missing or `state.txt` marks
# it absent; `--failed-out` and `--absent-out` name those tests.
#
# FAKE_FLAKES is a JSON list of {sha, test, phase, as, nth}: at that commit
# and phase the test reads as `as` (bad or ok), on the visits in `nth` or on
# every visit without it. FAKE_ABORT_AFTER=N exits 255 from the (N+1)th call
# on, FAKE_ABORT_CHECK exits 255 in the check phase, FAKE_NO_FAILED leaves
# `--failed-out` empty, FAKE_DIRTY=<phase> dirties a tracked file in that
# phase, and FAKE_HANG=<seconds> sleeps first, with a child that sleeps too.
_FAKE_STEP = '''\
import argparse, json, os, subprocess, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--tests", required=True)
p.add_argument("--failed-out", default=None)
p.add_argument("--absent-out", default=None)
args, _ = p.parse_known_args()
head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                      timeout=30, check=True).stdout.strip()
phase = "check" if args.failed_out else "bisect"
line = f"{head} {phase}"
log = Path(os.environ["FAKE_STEP_LOG"])
with log.open("a", encoding="utf-8") as fh:
    fh.write(line + "\\n")
calls = log.read_text(encoding="utf-8").splitlines()
hang = os.environ.get("FAKE_HANG")
if hang:
    child = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({hang})"])
    with open(os.environ["FAKE_PIDS"], "a", encoding="utf-8") as fh:
        fh.write(f"{os.getpid()}\\n{child.pid}\\n")
    time.sleep(float(hang))
    sys.exit(125)
abort_after = int(os.environ.get("FAKE_ABORT_AFTER", "0"))
if abort_after and len(calls) > abort_after:
    sys.exit(255)
if phase == "check" and os.environ.get("FAKE_ABORT_CHECK"):
    sys.exit(255)
state = dict(row.split("=", 1) for row in Path("state.txt").read_text(encoding="utf-8").split())
ids = [row for row in Path(args.tests).read_text(encoding="utf-8").split() if row]
visit = calls.count(line)
flakes = json.loads(os.environ.get("FAKE_FLAKES", "[]"))
failing, absent = [], []
for i in ids:
    name = i.split("::")[-1]
    if not Path(i.split("::")[0]).is_file() or state.get(name) == "absent":
        absent.append(i)
        continue
    bad = state.get(name) == "bad"
    for f in flakes:
        if f["sha"] == head and f["test"] == name and f["phase"] == phase and visit in f.get("nth", [visit]):
            bad = f["as"] == "bad"
    if bad:
        failing.append(i)
if args.failed_out:
    written = [] if os.environ.get("FAKE_NO_FAILED") else failing
    Path(args.failed_out).write_text("".join(i + "\\n" for i in written), encoding="utf-8")
if args.absent_out:
    Path(args.absent_out).write_text("".join(i + "\\n" for i in absent), encoding="utf-8")
if os.environ.get("FAKE_DIRTY") == phase:
    with open("state.txt", "a", encoding="utf-8") as fh:
        fh.write("dirty=1\\n")
sys.exit(1 if failing else 0)
'''

# The fake `uv`: `sync` passes unless FAKE_UV_SYNC_FAIL is set; `run
# --no-sync pytest --collect-only -q FILES` prints the `def test_*` ids of
# each file, and exits 2 when one holds `import BROKEN`; `run --no-sync
# pytest ... IDS` fails the ids `state.txt` marks bad and writes a JUnit
# report for `--junitxml=`. Each call is logged.
_FAKE_UV = '''\
import json, os, re, sys
from pathlib import Path
argv = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(argv) + "\\n")
if argv[0] == "sync":
    sys.exit(1 if os.environ.get("FAKE_UV_SYNC_FAIL") else 0)
args = [a for a in argv[4:] if not a.startswith("-") and a != "no:cacheprovider"]
if "--collect-only" in argv:
    broken = False
    for f in args:
        text = Path(f).read_text(encoding="utf-8")
        if "import BROKEN" in text:
            print(f"ERROR {f}")
            broken = True
            continue
        for name in re.findall(r"^def (test_\\w+)", text, re.M):
            print(f"{f}::{name}")
    sys.exit(2 if broken else 0)
state = dict(line.split("=", 1) for line in Path("state.txt").read_text(encoding="utf-8").split())
cases, red = [], 0
for i in args:
    path, name = i.split("::")
    body = ""
    if state.get(name) == "bad":
        body, red = '<failure message="boom">boom</failure>', red + 1
    cls = path[:-3].replace("/", ".")
    cases.append(f'<testcase classname="{cls}" name="{name}">{body}</testcase>')
for a in argv:
    if a.startswith("--junitxml="):
        Path(a.split("=", 1)[1]).write_text("<testsuites><testsuite>" + "".join(cases)
                                            + "</testsuite></testsuites>", encoding="utf-8")
sys.exit(1 if red else 0)
'''

# The fake `gh`: logs its arguments, then prints FAKE_GH_OUT's file or,
# with FAKE_GH_FAIL set, fails.
_FAKE_GH = '''\
import json, os, sys
from pathlib import Path
with open(os.environ["FAKE_GH_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
if os.environ.get("FAKE_GH_FAIL"):
    print("HTTP 502", file=sys.stderr)
    sys.exit(1)
print(Path(os.environ["FAKE_GH_OUT"]).read_text(encoding="utf-8"))
'''


# ----------------------------------------------------------- fixtures

def _git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=60, env=env, check=False,
    )


def _executable(path: Path, source: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


# Repositories built once per module, keyed by their history: building
# one costs a git call per step, and copying it is cheaper.
_Edit = tuple[int, str, "str | None"]
_Built = dict[tuple[tuple[str, ...], tuple[_Edit, ...]], tuple[Path, list[str]]]


class Env:
    """The sandbox: HOME, PATH with the fakes first, and their logs."""

    def __init__(self, root: Path, factory: pytest.TempPathFactory, built: _Built) -> None:
        self.root = root
        self.factory = factory
        self.built = built
        self.bin = root / "bin"
        self.step_log = root / "step.log"
        self.uv_log = root / "uv.log"
        self.gh_log = root / "gh.log"
        self.gh_out = root / "gh.json"
        self.fake_step = _executable(root / "fake_step.py", _FAKE_STEP)

    def repo(self, bad_tests: list[str], edits: tuple[_Edit, ...] = ()) -> list[str]:
        """A copy of the repository for `bad_tests` and `edits` at
        `root / "repo"`; see `_build_repo`. Returns its commit shas."""
        key = (tuple(bad_tests), edits)
        if key not in self.built:
            root = self.factory.mktemp("bisect-repo") / "repo"
            self.built[key] = (root, _build_repo(root, bad_tests, edits))
        root, shas = self.built[key]
        shutil.copytree(root, self.root / "repo", symlinks=True)
        return list(shas)

    @property
    def step_cmd(self) -> str:
        return shlex.join([sys.executable, str(self.fake_step)])

    def stepped(self) -> list[str]:
        """The commit of each step call, in order."""
        if not self.step_log.is_file():
            return []
        return [row.split()[0] for row in self.step_log.read_text(encoding="utf-8").splitlines()]

    def gh_calls(self) -> list[list[str]]:
        if not self.gh_log.is_file():
            return []
        return [json.loads(line) for line in self.gh_log.read_text(encoding="utf-8").splitlines()]

    def uv_calls(self) -> list[list[str]]:
        if not self.uv_log.is_file():
            return []
        return [json.loads(line) for line in self.uv_log.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def _built() -> _Built:
    return {}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory,
        _built: _Built) -> Env:
    home = tmp_path / "home"
    home.mkdir()
    e = Env(tmp_path, tmp_path_factory, _built)
    e.bin.mkdir()
    _executable(e.bin / "gh", _FAKE_GH)
    _executable(e.bin / "uv", _FAKE_UV)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(home / ".gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", f"{e.bin}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_STEP_LOG", str(e.step_log))
    monkeypatch.setenv("FAKE_UV_LOG", str(e.uv_log))
    monkeypatch.setenv("FAKE_GH_LOG", str(e.gh_log))
    monkeypatch.setenv("FAKE_GH_OUT", str(e.gh_out))
    for var in ("FAKE_ABORT_AFTER", "FAKE_ABORT_CHECK", "FAKE_FLAKES", "FAKE_NO_FAILED", "FAKE_DIRTY",
                "FAKE_HANG", "FAKE_UV_SYNC_FAIL", "FAKE_GH_FAIL"):
        monkeypatch.delenv(var, raising=False)
    return e


def _build_repo(root: Path, bad_tests: list[str], edits: tuple[_Edit, ...] = ()) -> list[str]:
    """One commit per entry, oldest first. Each entry names the tests (`a`
    to `d`) that fail at that commit; an upper-case letter marks that test
    absent, and `m` makes `tests/test_y.py` fail to import. Each edit is
    (commit, path, content) and writes that file at that commit, or deletes
    it when the content is None. Returns the commit shas, which the fixed
    dates make the same on every build."""
    home = root.parent / "home"
    home.mkdir(exist_ok=True)
    git_env = {**os.environ, "HOME": str(home), "GIT_CONFIG_GLOBAL": str(home / ".gitconfig"),
               "GIT_CONFIG_NOSYSTEM": "1"}
    root.mkdir()
    for args in (
        ["init", "-q", "-b", "main"],
        ["config", "user.name", "Nightly Test"],
        ["config", "user.email", "nightly@example.invalid"],
        ["config", "commit.gpgsign", "false"],
        ["config", "tag.gpgsign", "false"],
    ):
        assert _git(root, *args, env=git_env).returncode == 0
    (root / "tests").mkdir()
    source = "\n\n".join(f"def test_{t}():\n    pass\n" for t in "abcd")
    (root / "tests" / "test_x.py").write_text(source, encoding="utf-8")
    shas: list[str] = []
    for i, bad in enumerate(bad_tests):
        state = "".join(
            f"test_{t}={'absent' if t.upper() in bad else 'bad' if t in bad else 'ok'}\n" for t in "abcd"
        )
        (root / "state.txt").write_text(f"{state}n={i}\n", encoding="utf-8")
        broken = "import BROKEN\n\n\n" if "m" in bad else ""
        (root / "tests" / "test_y.py").write_text(f"{broken}def test_c():\n    pass\n", encoding="utf-8")
        for at, path, content in edits:
            if at != i:
                continue
            target = root / path
            if content is None:
                target.unlink()
            else:
                target.write_text(content, encoding="utf-8")
        assert _git(root, "add", "-A", env=git_env).returncode == 0
        date = f"2026-11-01T00:{i:02d}:00+00:00"
        r = _git(root, "commit", "-q", "-m", f"c{i}",
                 env={**git_env, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date})
        assert r.returncode == 0, r.stderr
        shas.append(_git(root, "rev-parse", "HEAD", env=git_env).stdout.strip())
    return shas


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _tests_file(path: Path, *ids: str) -> Path:
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    return path


Runner = Callable[..., tuple[int, dict[str, Any]]]


@pytest.fixture
def run_bisect(env: Env, tmp_path: Path) -> Runner:
    """`run` in-process; returns its exit status and result file."""

    def run(repo: Path, good: str, bad: str, *ids: str, points: dict[str, int | None] | None = None,
            extra: tuple[str, ...] = (), step: bool = True) -> tuple[int, dict[str, Any]]:
        out = tmp_path / "result.json"
        argv = ["run", "--repo-root", str(repo), "--good", good, "--bad", bad,
                "--tests", str(_tests_file(tmp_path / "ids.txt", *ids)), "--out", str(out)]
        if points is not None:
            argv += ["--points", str(_write(tmp_path / "points.json", points))]
        if step:
            argv += ["--step-cmd", env.step_cmd]
        rc = nb.main([*argv, *extra])
        return rc, json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}

    return run


# Commits 0-7. Test a breaks at 4, a commit inside PR 12, whose merge
# commit is 5. PR 11 merged at 2 and PR 13 at 7.
_LINEAR = ["", "", "", "", "a", "a", "a", "a"]


def _linear_points(shas: list[str]) -> dict[str, int | None]:
    return {shas[2]: 11, shas[5]: 12, shas[7]: 13}


# --------------------------------------------------------------- bisect

def test_culprit_is_the_merge_point_and_other_commits_never_run(
    env: Env, tmp_path: Path, run_bisect: Runner,
) -> None:
    shas = env.repo(_LINEAR)
    points = _linear_points(shas)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=points)
    assert rc == 0
    assert result["events"] == [{"key": "pr-12", "kind": "pr", "pr": 12, "sha": shas[5], "tests": [_A]}]
    stepped = set(env.stepped())
    assert shas[5] in stepped
    assert stepped <= {shas[0], *points}, "a commit that is not a merge point was tested"


def test_single_commit_prs(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(["", "", "a", "a"])
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[3], _A,
                            points={shas[1]: 21, shas[2]: 22, shas[3]: 23})
    assert rc == 0
    assert [e["key"] for e in result["events"]] == ["pr-22"]
    assert result["events"][0]["sha"] == shas[2]


def test_two_culprits_get_one_event_each(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(["", "", "a", "a", "ab", "ab"])
    points = {shas[i]: 30 + i for i in range(1, 6)}
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[5], _A, _B, points=points)
    assert rc == 0
    assert [(e["key"], e["tests"]) for e in result["events"]] == [("pr-32", [_A]), ("pr-34", [_B])]
    assert [r["good"] for r in result["rounds"]] == [shas[0], shas[2]]


def test_without_merge_points_every_commit_is_tested(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(_LINEAR)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points={})
    assert rc == 0
    assert result["events"] == [{"key": f"commit-{shas[4][:12]}", "kind": "commit", "sha": shas[4], "tests": [_A]}]
    assert set(env.stepped()) - {shas[0], shas[2], shas[5], shas[7]}, "no commit outside a PR merge was tested"


def test_past_the_budget_reports_the_range(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(_LINEAR)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas),
                            extra=("--budget-seconds", "0"))
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[7], [_A], nb.REASON_BUDGET)]
    assert env.stepped() == []


def test_a_step_that_aborts_reports_the_range_so_far(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("FAKE_ABORT_AFTER", "1")
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 1
    first = env.stepped()[0]
    # The one step that ran marked its commit good (2) or bad (5).
    good, bad = (first, shas[7]) if first == shas[2] else (shas[0], first)
    assert first in (shas[2], shas[5])
    assert result["events"] == [nb.event_for_range(good, bad, [_A], nb.REASON_INCONCLUSIVE)]


def test_a_culprit_that_passes_when_checked_again_is_a_range(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("FAKE_FLAKES", json.dumps(
        [{"sha": shas[5], "test": "test_a", "phase": "check", "as": "ok"}]))
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[7], [_A], nb.REASON_UNVERIFIED)]
    assert result["rounds"][0]["verify"]["culprit_rc"] == nb.STEP_GOOD


def test_the_bisect_is_reset_after_the_run(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(["", "", "a", "a"])
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[3], _A,
                            points={shas[1]: 21, shas[2]: 22, shas[3]: 23})
    assert rc == 0
    assert result["rounds"][0]["log"], "the round kept no bisect log"
    worktree = tmp_path / "repo" / ".bisect"
    assert worktree.is_dir()
    log = _git(worktree, "bisect", "log")
    assert log.returncode != 0, f"a bisect is still in progress:\n{log.stdout}"


def test_a_second_run_replaces_the_worktree(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(["", "", "a", "a"])
    points: dict[str, int | None] = {shas[1]: 21, shas[2]: 22, shas[3]: 23}
    assert run_bisect(tmp_path / "repo", shas[0], shas[3], _A, points=points)[0] == 0
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[3], _A, points=points)
    assert rc == 0
    assert [e["key"] for e in result["events"]] == ["pr-22"]


def test_the_default_step_runs_by_absolute_path(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    """No `--step-cmd`: `git bisect run` runs this script's `step` inside
    the worktree, which has no `scripts/`, under the fake `uv`."""
    shas = env.repo(["", "", "a", "a"])
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[3], _A,
                            points={shas[1]: 21, shas[2]: 22, shas[3]: 23}, step=False)
    assert rc == 0, result
    assert [e["key"] for e in result["events"]] == ["pr-22"]
    assert any(call[0] == "sync" for call in env.uv_calls())
    cmd = nb.default_step_cmd()
    assert Path(cmd[1]).is_absolute()
    assert Path(cmd[1]) == (_REPO / "scripts" / "nightly_bisect.py").resolve()
    assert cmd[2] == "step"


@pytest.mark.parametrize("module_id", ["tests.test_y", "::tests.test_y"])
def test_a_module_that_stops_importing_is_bisected_by_its_file(
    env: Env, tmp_path: Path, run_bisect: Runner, module_id: str,
) -> None:
    """pytest's JUnit names a module that fails to import by its dotted
    path alone. The step judges that id by whether the module's file
    collects, so the culprit is the merge that broke the import."""
    shas = env.repo(["", "", "", "m", "m", "m"])
    points: dict[str, int | None] = {shas[1]: 41, shas[3]: 42, shas[5]: 43}
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[5], module_id, points=points, step=False)
    assert rc == 0, result
    assert result["events"] == [{"key": "pr-42", "kind": "pr", "pr": 42, "sha": shas[3], "tests": [module_id]}]


def test_dry_run_changes_nothing(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(_LINEAR)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas),
                            extra=("--dry-run",))
    assert (rc, result) == (0, {})
    assert env.stepped() == []
    assert not (tmp_path / "repo" / ".bisect").exists()
    assert _git(tmp_path / "repo", "worktree", "list").stdout.count("\n") == 1


@pytest.mark.parametrize(
    "case",
    ["not-an-ancestor", "same-commit", "unknown-sha", "no-tests", "missing-tests", "bad-points"],
)
def test_run_usage_errors_exit_2(env: Env, tmp_path: Path, case: str) -> None:
    shas = env.repo(["", "a"])
    ids = _tests_file(tmp_path / "ids.txt", _A)
    good, bad = shas[0], shas[1]
    extra: list[str] = []
    if case == "not-an-ancestor":
        good, bad = bad, good
    elif case == "same-commit":
        good = bad
    elif case == "unknown-sha":
        good = "0" * 40
    elif case == "no-tests":
        ids = _tests_file(tmp_path / "ids.txt")
    elif case == "missing-tests":
        ids = tmp_path / "absent.txt"
    else:
        extra = ["--points", str(_write(tmp_path / "p.json", {shas[1]: "twelve"}))]
    rc = nb.main(["run", "--repo-root", str(tmp_path / "repo"), "--good", good, "--bad", bad,
                  "--tests", str(ids), "--out", str(tmp_path / "r.json"), "--step-cmd", env.step_cmd, *extra])
    assert rc == 2
    assert not (tmp_path / "r.json").exists()
    assert env.stepped() == []


# ---------------------------------------------------------- merge points

def _merge_points(repo: Path, good: str, bad: str, out: Path, *extra: str) -> int:
    return nb.main(["merge-points", "--repo-root", str(repo), "--good", good, "--bad", bad,
                    "--out", str(out), *extra])


def test_merge_points_keeps_merges_in_the_range_and_bad(env: Env, tmp_path: Path) -> None:
    shas = env.repo(_LINEAR)
    _write(env.gh_out, [
        {"number": 13, "mergeCommit": {"oid": shas[7]}, "title": "t", "url": "u"},
        {"number": 12, "mergeCommit": {"oid": shas[5]}, "title": "t", "url": "u"},
        {"number": 11, "mergeCommit": {"oid": shas[2]}, "title": "t", "url": "u"},
        {"number": 9, "mergeCommit": {"oid": shas[0]}, "title": "before the range", "url": "u"},
        {"number": 99, "mergeCommit": {"oid": "f" * 40}, "title": "elsewhere", "url": "u"},
        {"number": 100, "mergeCommit": None, "title": "no merge commit", "url": "u"},
    ])
    out = tmp_path / "points.json"
    assert _merge_points(tmp_path / "repo", shas[0], shas[6], out, "--repo", "o/r") == 0
    points = json.loads(out.read_text(encoding="utf-8"))
    assert list(points.items()) == [(shas[2], 11), (shas[5], 12), (shas[6], None)]
    # The good commit is dated 2026-11-01 UTC; the search starts a day earlier.
    assert env.gh_calls() == [[
        "pr", "list", "--state", "merged", "--search", "merged:>=2026-10-31",
        "--json", "number,mergeCommit,title,url", "--limit", "500", "--repo", "o/r",
    ]]


def test_merge_points_api_failure_means_every_commit(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("FAKE_GH_FAIL", "1")
    out = tmp_path / "points.json"
    assert _merge_points(tmp_path / "repo", shas[0], shas[7], out) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {}
    assert len(env.gh_calls()) == 1
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points={})
    assert rc == 0
    assert result["events"][0]["key"] == f"commit-{shas[4][:12]}"


def test_merge_points_dry_run_calls_no_api(env: Env, tmp_path: Path) -> None:
    shas = env.repo(_LINEAR)
    out = tmp_path / "points.json"
    assert _merge_points(tmp_path / "repo", shas[0], shas[7], out, "--dry-run") == 0
    assert env.gh_calls() == []
    assert not out.exists()


def test_merge_points_unknown_sha_exits_2(env: Env, tmp_path: Path) -> None:
    shas = env.repo(_LINEAR)
    assert _merge_points(tmp_path / "repo", "0" * 40, shas[7], tmp_path / "p.json") == 2
    assert env.gh_calls() == []


# ------------------------------------------------------------------ step

@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"past_deadline": True}, nb.STEP_ABORT),
        ({"past_deadline": True, "present": [_A], "run_rc": 0}, nb.STEP_ABORT),
        ({"past_deadline": False, "sync_rc": 1}, nb.STEP_SKIP),
        ({"past_deadline": False, "sync_rc": None}, nb.STEP_SKIP),
        ({"past_deadline": False, "collect_rc": 2, "present": [_A]}, nb.STEP_SKIP),
        ({"past_deadline": False, "collect_rc": None}, nb.STEP_SKIP),
        ({"past_deadline": False, "collect_rc": 5}, nb.STEP_GOOD),
        ({"past_deadline": False, "present": []}, nb.STEP_GOOD),
        ({"past_deadline": False, "present": [_A], "run_rc": 0}, nb.STEP_GOOD),
        ({"past_deadline": False, "present": [_A], "run_rc": 1}, nb.STEP_BAD),
        ({"past_deadline": False, "present": [_A], "run_rc": 2}, nb.STEP_SKIP),
        ({"past_deadline": False, "present": [_A], "run_rc": 4}, nb.STEP_SKIP),
        ({"past_deadline": False, "present": [_A], "run_rc": None}, nb.STEP_SKIP),
        ({"past_deadline": False, "module_rcs": [2]}, nb.STEP_BAD),
        ({"past_deadline": False, "module_rcs": [0, 2], "present": [_A], "run_rc": 0}, nb.STEP_BAD),
        ({"past_deadline": False, "module_rcs": [0, 5]}, nb.STEP_GOOD),
        ({"past_deadline": False, "module_rcs": [0], "present": [_A], "run_rc": 1}, nb.STEP_BAD),
        ({"past_deadline": False, "module_rcs": [None]}, nb.STEP_SKIP),
        ({"past_deadline": False, "module_rcs": [4]}, nb.STEP_SKIP),
        ({"past_deadline": True, "module_rcs": [2]}, nb.STEP_ABORT),
        ({"past_deadline": False, "sync_rc": 1, "module_rcs": [2]}, nb.STEP_SKIP),
    ],
    ids=["deadline", "deadline-first", "sync-fails", "sync-timeout", "collect-error", "collect-timeout",
         "nothing-collected", "no-id-present", "run-passes", "run-fails", "run-interrupted",
         "run-usage", "run-timeout", "module-error", "module-error-wins", "module-collects",
         "module-collects-run-fails", "module-timeout", "module-usage", "module-deadline", "module-sync"],
)
def test_step_exit_mapping(kwargs: dict[str, Any], status: int) -> None:
    assert nb.step_verdict(**kwargs) == status


def _step(tmp_path: Path, worktree: Path, *extra: str) -> int:
    """The real step as `git bisect run` runs it: a child in the worktree."""
    r = subprocess.run(
        [*nb.default_step_cmd(), "--tests", str(tmp_path / "ids.txt"), *extra],
        cwd=worktree, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60, check=False,
    )
    return r.returncode


def test_step_past_the_deadline_aborts_without_running(env: Env, tmp_path: Path) -> None:
    env.repo(["a"])
    _tests_file(tmp_path / "ids.txt", _A)
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "0") == nb.STEP_ABORT
    assert env.uv_calls() == []


def test_step_sync_failure_skips(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env.repo(["a"])
    _tests_file(tmp_path / "ids.txt", _A)
    monkeypatch.setenv("FAKE_UV_SYNC_FAIL", "1")
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12") == nb.STEP_SKIP
    assert [c[0] for c in env.uv_calls()] == ["sync"]


def test_step_with_no_test_present_is_good(env: Env, tmp_path: Path) -> None:
    env.repo(["a"])
    _tests_file(tmp_path / "ids.txt", "tests/test_new.py::test_a", "tests/test_x.py::test_gone")
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12") == nb.STEP_GOOD
    # test_new.py does not exist, so only test_x.py is collected, and
    # test_gone is not in it: nothing is run.
    assert [c[:5] for c in env.uv_calls()] == [
        ["sync", "--frozen"], ["run", "--no-sync", "pytest", "-p", "no:cacheprovider"],
    ]
    assert "--collect-only" in env.uv_calls()[1]


def test_step_names_the_failing_tests(env: Env, tmp_path: Path) -> None:
    env.repo(["a"])
    _tests_file(tmp_path / "ids.txt", _A, _B, "tests/test_x.py::test_gone")
    failed = tmp_path / "failed.txt"
    status = _step(tmp_path, tmp_path / "repo", "--deadline", "9e12", "--failed-out", str(failed),
                   "--python", "3.13", "--extra", "archive")
    assert status == nb.STEP_BAD
    assert failed.read_text(encoding="utf-8") == f"{_A}\n"
    sync, _, run = env.uv_calls()
    assert sync == ["sync", "--frozen", "--extra", "archive", "--python", "3.13"]
    assert run[-2:] == [_A, _B]


@pytest.mark.parametrize(("bad_tests", "status", "failed"), [
    (["m"], nb.STEP_BAD, "tests.test_y\n"),
    ([""], nb.STEP_GOOD, ""),
])
def test_step_judges_a_module_id_by_its_file(
    env: Env, tmp_path: Path, bad_tests: list[str], status: int, failed: str,
) -> None:
    env.repo(bad_tests)
    _tests_file(tmp_path / "ids.txt", "tests.test_y", _B)
    out = tmp_path / "failed.txt"
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12", "--failed-out", str(out)) == status
    assert out.read_text(encoding="utf-8") == failed
    collects = [c[-1] for c in env.uv_calls() if "--collect-only" in c]
    assert collects == ["tests/test_y.py", "tests/test_x.py"]


def test_step_with_an_absent_module_file_is_good(env: Env, tmp_path: Path) -> None:
    env.repo(["m"])
    _tests_file(tmp_path / "ids.txt", "tests.test_gone")
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12") == nb.STEP_GOOD
    assert [c[0] for c in env.uv_calls()] == ["sync"]


def test_step_that_cannot_read_its_input_aborts(env: Env, tmp_path: Path) -> None:
    env.repo(["a"])
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12") == nb.STEP_ABORT


def test_step_dry_run_runs_nothing(env: Env, tmp_path: Path) -> None:
    env.repo(["a"])
    _tests_file(tmp_path / "ids.txt", _A)
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12", "--dry-run") == 0
    assert env.uv_calls() == []


# ------------------------------------------------------------ pure parts

@pytest.mark.parametrize(
    ("node_id", "key"),
    [
        ("tests/test_a.py::test_x", "tests.test_a::test_x"),
        ("tests/sub/test_a.py::TestC::test_x", "tests.sub.test_a.TestC::test_x"),
        ("tests/test_a.py::TestC::TestD::test_x[1.5-a b]", "tests.test_a.TestC.TestD::test_x[1.5-a b]"),
        ("tests/test_a.py::test_x[a::b]", "tests.test_a::test_x[a::b]"),
    ],
)
def test_junit_key(node_id: str, key: str) -> None:
    assert nb.junit_key(node_id) == key


@pytest.mark.parametrize(
    ("test_id", "path"),
    [
        ("tests.test_x", "tests/test_x.py"),
        ("::tests.sub.test_x", "tests/sub/test_x.py"),
        ("tests/test_x.py", "tests/test_x.py"),
        ("tests/test_x.py::test_a", None),
        ("tests/test_x.py::TestC::test_a[a.b]", None),
        ("tests.test_x::test_a", None),
        ("::", None),
        ("tests.test-x", None),
    ],
)
def test_module_file(test_id: str, path: str | None) -> None:
    assert nb.module_file(test_id) == path


def test_test_files_leaves_module_ids_out() -> None:
    assert nb.test_files(["tests.test_y", _A, "tests/test_z.py"]) == ["tests/test_x.py"]


def test_failing_ids_reads_failures_and_errors() -> None:
    report = (
        '<testsuites><testsuite>'
        '<testcase classname="tests.test_a" name="test_ok"/>'
        '<testcase classname="tests.test_a" name="test_fail"><failure message="x"/></testcase>'
        '<testcase classname="tests.test_a.TestC" name="test_err[p]"><error message="x"/></testcase>'
        '</testsuite></testsuites>'
    )
    ids = ["tests/test_a.py::test_ok", "tests/test_a.py::TestC::test_err[p]", "tests/test_a.py::test_fail"]
    assert nb.failing_ids(report, ids) == ids[1:]


def test_collected_and_present_ids() -> None:
    listing = "tests/test_a.py::test_x\ntests/test_a.py::test_y[1]\n\n2 tests collected in 0.01s\n"
    collected = nb.collected_ids(listing)
    assert collected == {"tests/test_a.py::test_x", "tests/test_a.py::test_y[1]"}
    assert nb.present_ids(["tests/test_a.py::test_y[1]", "tests/test_a.py::test_z"], collected) == [
        "tests/test_a.py::test_y[1]",
    ]
    assert nb.test_files(["a.py::x", "b.py::C::y", "a.py::z"]) == ["a.py", "b.py"]


def test_read_outcome() -> None:
    order = ["c1", "c2", "c3", "c4", "c5"]
    points = {"c2", "c4", "c5"}
    # c2 good, c4 bad: c4 is the culprit.
    assert nb.read_outcome(order, points, "c0", "c4", ["c2"]) == nb.Outcome("c4", "c2", "c4")
    # c4 bad, c2 untested (skipped): the range is c0..c4.
    assert nb.read_outcome(order, points, "c0", "c4", []) == nb.Outcome(None, "c0", "c4")
    # The first point is bad: the culprit, with `lo` before it.
    assert nb.read_outcome(order, points, "c0", "c2", []) == nb.Outcome("c2", "c0", "c2")
    # Without merge-point data, every commit is one.
    assert nb.read_outcome(order, None, "c0", "c3", ["c2"]) == nb.Outcome("c3", "c2", "c3")
    assert nb.read_outcome(order, None, "c0", "c3", ["c1"]) == nb.Outcome(None, "c1", "c3")


def test_skip_list_and_points_between() -> None:
    assert nb.skip_list(["a", "b", "c"], {"b", "c"}) == ["a"]
    assert nb.skip_list(["a", "b", "c"], None) == []
    assert nb.points_between(["a", "b", "c"], {"c"}, "c") == []
    assert nb.points_between(["a", "b", "c"], None, "c") == ["a", "b"]


def test_merged_since_is_the_utc_day_before() -> None:
    # 2026-11-04T00:30:00Z, a commit at 09:30 local time in UTC+9.
    assert nb.merged_since(1793752200) == "2026-11-03"


def test_add_event_merges_one_key() -> None:
    events: list[dict[str, Any]] = []
    nb.add_event(events, nb.event_for_range("a" * 40, "b" * 40, ["t1"], nb.REASON_ROUNDS))
    nb.add_event(events, nb.event_for_range("a" * 40, "b" * 40, ["t1", "t2"], nb.REASON_CAP))
    assert [(e["key"], e["tests"]) for e in events] == [(f"range-{'a' * 12}-{'b' * 12}", ["t1", "t2"])]


def test_parse_tests_and_points() -> None:
    assert nb.parse_tests("# c\n\n a::b \na::b\nc::d\n") == ["a::b", "c::d"]
    assert nb.parse_points({"s": 1, "t": None}) == {"s": 1, "t": None}
    for bad in ([], {"s": "1"}, {"s": True}):
        with pytest.raises(ValueError):
            nb.parse_points(bad)
