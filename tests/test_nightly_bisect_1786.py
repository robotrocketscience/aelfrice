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
import signal
import stat
import subprocess
import sys
import tempfile
import time
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
# every visit without it. FAKE_EXITS is a JSON list of {sha, phase, rc, nth}:
# at that commit and phase the step exits rc. FAKE_ABORT_AFTER=N exits 255 from the (N+1)th call
# on, FAKE_ABORT_CHECK exits 255 in the check phase (given a sha, at that
# commit only), FAKE_NO_FAILED leaves
# `--failed-out` empty, FAKE_DIRTY=<phase> dirties a tracked file in that
# phase, FAKE_HANG=<seconds> sleeps first, with a child that sleeps too, and
# FAKE_READONLY=<path> leaves in the check phase a read-only folder and a
# symbolic link to that path. FAKE_RM_GIT=<phase> deletes the worktree's `.git`
# file in that phase. FAKE_CEILING_LOG names a file for GIT_CEILING_DIRECTORIES.
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
with open(os.environ["FAKE_ENV_LOG"], "a", encoding="utf-8") as fh:
    fh.write(f"step {phase} {os.environ.get('GIT_DIR')}\\n")
with open(os.environ["FAKE_CEILING_LOG"], "a", encoding="utf-8") as fh:
    fh.write(f"step {phase} {os.environ.get('GIT_CEILING_DIRECTORIES')}\\n")
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
if phase == "check" and os.environ.get("FAKE_ABORT_CHECK") in ("1", head):
    sys.exit(255)
state = dict(row.split("=", 1) for row in Path("state.txt").read_text(encoding="utf-8").split())
ids = [row for row in Path(args.tests).read_text(encoding="utf-8").split() if row]
visit = calls.count(line)
flakes = json.loads(os.environ.get("FAKE_FLAKES", "[]"))
for e in json.loads(os.environ.get("FAKE_EXITS", "[]")):
    if e["sha"] == head and e["phase"] == phase and visit in e.get("nth", [visit]):
        sys.exit(e["rc"])
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
if os.environ.get("FAKE_RM_GIT") == phase:
    Path(".git").unlink(missing_ok=True)
if os.environ.get("FAKE_READONLY") and phase == "check" and not Path("link").is_symlink():
    Path("link").symlink_to(os.environ["FAKE_READONLY"])
    Path("locked").mkdir()
    Path("locked/file.txt").write_text("x", encoding="utf-8")
    Path("locked").chmod(0o555)
sys.exit(1 if failing else 0)
'''

# The fake `uv`: `sync` passes unless FAKE_UV_SYNC_FAIL is set; `run
# --no-sync pytest --collect-only -q FILES` prints the `def test_*` ids of
# each file, and exits 2 when one holds `import BROKEN`, or 4 when a folder's
# `conftest.py` does; `run --no-sync pytest ... IDS` fails the ids `state.txt`
# marks bad and writes a JUnit report for `--junitxml=`, which
# FAKE_UV_JUNIT=missing leaves out and FAKE_UV_JUNIT=corrupt makes unreadable.
# Each call is logged. FAKE_UV_HANG=<stage> makes that stage (sync, collect
# or run) sleep with a child that sleeps too.
_FAKE_UV = '''\
import json, os, re, subprocess, sys, time
from pathlib import Path
argv = sys.argv[1:]
with open(os.environ["FAKE_UV_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(argv) + "\\n")
stage = "sync" if argv[0] == "sync" else "collect" if "--collect-only" in argv else "run"
with open(os.environ["FAKE_ENV_LOG"], "a", encoding="utf-8") as fh:
    fh.write(f"uv {stage} {os.environ.get('GIT_DIR')}\\n")
with open(os.environ["FAKE_CEILING_LOG"], "a", encoding="utf-8") as fh:
    fh.write(f"uv {stage} {os.environ.get('GIT_CEILING_DIRECTORIES')}\\n")
if os.environ.get("FAKE_UV_HANG") == stage:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    with open(os.environ["FAKE_PIDS"], "a", encoding="utf-8") as fh:
        fh.write(f"{os.getpid()}\\n{child.pid}\\n")
    time.sleep(30)
    sys.exit(0)
if argv[0] == "sync":
    sys.exit(1 if os.environ.get("FAKE_UV_SYNC_FAIL") else 0)
args = [a for a in argv[4:] if not a.startswith("-") and a != "no:cacheprovider"]
if "--collect-only" in argv:
    status = 0
    for f in args:
        if f.endswith("/"):
            if "import BROKEN" in Path(f, "conftest.py").read_text(encoding="utf-8"):
                status = 4
            continue
        text = Path(f).read_text(encoding="utf-8")
        if "import BROKEN" in text:
            print(f"ERROR {f}")
            status = 2
            continue
        for name in re.findall(r"^def (test_\\w+)", text, re.M):
            print(f"{f}::{name}")
    sys.exit(status)
state = dict(line.split("=", 1) for line in Path("state.txt").read_text(encoding="utf-8").split())
cases, red = [], 0
for i in args:
    path, name = i.split("::")
    body = ""
    if state.get(name) == "bad":
        body, red = '<failure message="boom">boom</failure>', red + 1
    cls = path[:-3].replace("/", ".")
    cases.append(f'<testcase classname="{cls}" name="{name}">{body}</testcase>')
report = "<testsuites><testsuite>" + "".join(cases) + "</testsuite></testsuites>"
mode = os.environ.get("FAKE_UV_JUNIT")
for a in argv:
    if a.startswith("--junitxml=") and mode != "missing":
        Path(a.split("=", 1)[1]).write_text("<testsuites" if mode == "corrupt" else report, encoding="utf-8")
sys.exit(1 if red else 0)
'''

# The fake `gh`: logs its arguments, its folder and its GIT_DIR, then prints FAKE_GH_OUT's file. With
# FAKE_GH_FAIL set it fails; set to `with-output`, it prints the file first.
_FAKE_GH = '''\
import json, os, sys
from pathlib import Path
with open(os.environ["FAKE_GH_LOG"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(sys.argv[1:]) + "\\n")
with open(os.environ["FAKE_ENV_LOG"], "a", encoding="utf-8") as fh:
    fh.write(f"gh {os.getcwd()} {os.environ.get('GIT_DIR')}\\n")
if os.environ.get("FAKE_GH_FAIL"):
    if os.environ["FAKE_GH_FAIL"] == "with-output":
        print(Path(os.environ["FAKE_GH_OUT"]).read_text(encoding="utf-8"))
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
        self.pids_file = root / "pids.txt"
        self.env_log = root / "env.log"
        self.ceiling_log = root / "ceiling.log"
        self.tmp = root / "tmp"
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

    def folders(self) -> list[Path]:
        """The folders `run` made and has not removed."""
        return sorted(self.tmp.glob("nightly-bisect-*"))

    def pids(self) -> list[int]:
        """The processes the hanging fakes started and named."""
        if not self.pids_file.is_file():
            return []
        return [int(row) for row in self.pids_file.read_text(encoding="utf-8").split()]

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
    e.tmp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(e.tmp))
    monkeypatch.setenv("TMPDIR", str(e.tmp))
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
    monkeypatch.setenv("FAKE_PIDS", str(e.pids_file))
    monkeypatch.setenv("FAKE_ENV_LOG", str(e.env_log))
    monkeypatch.setenv("FAKE_CEILING_LOG", str(e.ceiling_log))
    for var in ("FAKE_ABORT_AFTER", "FAKE_ABORT_CHECK", "FAKE_EXITS", "FAKE_FLAKES", "FAKE_NO_FAILED", "FAKE_DIRTY",
                "FAKE_HANG", "FAKE_READONLY", "FAKE_RM_GIT", "FAKE_UV_HANG", "FAKE_UV_JUNIT", "FAKE_UV_SYNC_FAIL", "FAKE_GH_FAIL"):
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
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
        assert _git(root, "add", "-A", env=git_env).returncode == 0
        date = f"2026-11-01T00:{i:02d}:00+00:00"
        r = _git(root, "commit", "-q", "-m", f"c{i}",
                 env={**git_env, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date})
        assert r.returncode == 0, r.stderr
        shas.append(_git(root, "rev-parse", "HEAD", env=git_env).stdout.strip())
    return shas


def _running(pid: int) -> bool:
    """True when `pid` is a live process; a zombie has ended."""
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=30, check=False).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _survivors(pids: list[int]) -> list[int]:
    """The processes in `pids` still running after a few seconds to end."""
    assert pids, "the fake never started, so this proves nothing"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and any(_running(p) for p in pids):
        time.sleep(0.1)
    return [p for p in pids if _running(p)]


def _write(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _tests_file(path: Path, *ids: str) -> Path:
    path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    return path


Runner = Callable[..., tuple[int, dict[str, Any]]]


@pytest.fixture
def run_bisect(env: Env, tmp_path: Path) -> Runner:
    """`run` in-process; returns its exit status and result file. Every run
    removes the folder it made."""

    def run(repo: Path, good: str, bad: str, *ids: str, points: dict[str, int | None] | None = None,
            extra: tuple[str, ...] = (), step: bool = True) -> tuple[int, dict[str, Any]]:
        out = tmp_path / "result.json"
        argv = ["run", "--repo-root", str(repo), "--good", good, "--bad", bad,
                "--tests", str(_tests_file(tmp_path / "ids.txt", *ids)), "--out", str(out)]
        if points is not None:
            argv += ["--points", str(_write(tmp_path / "points.json", points))]
        if step:
            argv += ["--step-cmd", env.step_cmd]
        before = env.folders()
        rc = nb.main([*argv, *extra])
        assert env.folders() == before, "the run left its folder"
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
    assert result["events"] == [
        {"key": "pr-12", "kind": "pr", "pr": 12, "sha": shas[5], "preceding": shas[2], "tests": [_A]},
    ]
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
    assert result["events"] == [
        {"key": f"commit-{shas[4][:12]}", "kind": "commit", "sha": shas[4], "preceding": shas[3], "tests": [_A]},
    ]
    assert set(env.stepped()) - {shas[0], shas[2], shas[5], shas[7]}, "no commit outside a PR merge was tested"


def test_past_the_budget_reports_the_range(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(_LINEAR)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas),
                            extra=("--budget-seconds", "0"))
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[7], [_A], nb.REASON_BUDGET)]
    assert env.stepped() == []


def _merge_a_side_branch(repo: Path, fork: str) -> str:
    """Merge a branch from `fork` into main. Returns the merge commit."""
    for args in (["checkout", "-q", "-b", "side", fork], ["commit", "-q", "--allow-empty", "-m", "side"],
                 ["checkout", "-q", "main"], ["merge", "-q", "--no-ff", "--no-edit", "side"]):
        r = _git(repo, *args)
        assert r.returncode == 0, r.stderr
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


@pytest.mark.parametrize("fork_at", [0, 1], ids=["before-good", "after-good"])
def test_a_range_that_is_not_one_chain_tests_nothing_and_gets_a_range(
    env: Env, tmp_path: Path, run_bisect: Runner, fork_at: int,
) -> None:
    """A side commit forked before `--good` raised `ValueError` with no result,
    and the commit before a merge point was its list neighbour, not a parent."""
    shas = env.repo(["", "", "a"])
    repo = tmp_path / "repo"
    merge = _merge_a_side_branch(repo, shas[fork_at])
    rc, result = run_bisect(repo, shas[1], merge, _A, "tests/test_gone.py::test_a")
    assert rc == 1
    assert [e["reasons"] for e in result["events"]] == [
        {_A: nb.REASON_NONLINEAR, "tests/test_gone.py::test_a": nb.REASON_UNMAPPED},
    ]
    assert (result["events"][0]["good"], result["events"][0]["bad"]) == (shas[1], merge)
    assert result["complete"] is True
    assert env.stepped() == []
    assert _git(repo, "worktree", "list").stdout.count("\n") == 1


def test_a_dry_run_of_a_range_that_is_not_one_chain_says_so(
    env: Env, tmp_path: Path, run_bisect: Runner, capsys: pytest.CaptureFixture[str],
) -> None:
    shas = env.repo(["", "", "a"])
    merge = _merge_a_side_branch(tmp_path / "repo", shas[0])
    rc, _ = run_bisect(tmp_path / "repo", shas[1], merge, _A, extra=("--dry-run",))
    assert rc == 0
    assert "not one chain" in capsys.readouterr().out


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


def test_a_bisect_the_deadline_stopped_reads_as_budget(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("FAKE_ABORT_AFTER", "1")
    real = nb.Bisector.bisect_round

    def then_out_of_time(self: nb.Bisector, lo: str, tests: list[str], n: int) -> Any:
        out = real(self, lo, tests, n)
        self.deadline = 0.0
        return out

    monkeypatch.setattr(nb.Bisector, "bisect_round", then_out_of_time)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 1
    assert [set(e["reasons"].values()) for e in result["events"]] == [{nb.REASON_BUDGET}]


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


def _timeout_once_the_fake_hangs(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wait on a long-running child reads as its timeout once the hanging
    fake has started, so these tests depend on neither the clock nor the load."""
    real_wait = subprocess.Popen.wait

    def wait(self: subprocess.Popen[Any], timeout: float | None = None) -> int:
        if timeout is None or timeout <= nb._TERM_GRACE_SECONDS or not isinstance(self.args, list):
            return real_wait(self, timeout)
        if self.args[:3] == ["git", "bisect", "run"] or "--failed-out" in self.args:
            limit = time.monotonic() + 60
            while not env.pids() and time.monotonic() < limit:
                time.sleep(0.05)
            raise subprocess.TimeoutExpired(self.args, timeout)
        return real_wait(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)


def test_a_hung_bisect_ends_with_nothing_left_running(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budget 10 and step timeout 3 once took 73.5 s and left 3 processes."""
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("FAKE_HANG", "30")
    _timeout_once_the_fake_hangs(env, monkeypatch)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 1
    assert "timed out" in result["rounds"][0]["output"]
    assert result["events"] == [nb.event_for_range(shas[0], shas[7], [_A], nb.REASON_INCONCLUSIVE)]
    assert _survivors(env.pids()) == []


def test_a_hung_recheck_ends_with_nothing_left_running(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(["", "a"])
    monkeypatch.setenv("FAKE_HANG", "30")
    _timeout_once_the_fake_hangs(env, monkeypatch)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[1], _A, points={shas[1]: 7})
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[1], [_A], nb.REASON_UNVERIFIED)]
    assert result["rounds"][0]["verify"]["culprit_rc"] == nb.STEP_SKIP
    assert _survivors(env.pids()) == []


def test_a_deadline_during_the_recheck_keeps_the_narrowed_range(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bisect had finished, yet the result named the whole range."""
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("FAKE_ABORT_CHECK", "1")
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[2], shas[5], [_A], nb.REASON_BUDGET)]


def test_a_bystander_that_flakes_once_at_the_culprit_goes_to_the_next_round(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test b breaks at 4, but failed once when the culprit of test a ran."""
    shas = env.repo(["", "", "a", "a", "ab", "ab"])
    monkeypatch.setenv("FAKE_FLAKES", json.dumps(
        [{"sha": shas[2], "test": "test_b", "phase": "check", "as": "bad", "nth": [1]}]))
    points = {shas[i]: 30 + i for i in range(1, 6)}
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[5], _A, _B, points=points)
    assert rc == 0
    assert [(e["key"], e["tests"]) for e in result["events"]] == [("pr-32", [_A]), ("pr-34", [_B])]


def test_a_test_that_fails_at_the_preceding_point_is_not_blamed_on_the_culprit(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test b reads good during the bisect and fails when the point before
    the culprit runs again. Test a is still the culprit's."""
    shas = env.repo(["", "", "ab", "ab"])
    monkeypatch.setenv("FAKE_FLAKES", json.dumps(
        [{"sha": shas[1], "test": "test_b", "phase": "check", "as": "bad", "nth": [1]}]))
    points = {shas[i]: 30 + i for i in range(1, 4)}
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[3], _A, _B, points=points)
    assert rc == 1
    assert result["events"] == [
        nb.event_for_range(shas[0], shas[2], [_B], nb.REASON_UNVERIFIED),
        nb.event_for_culprit(shas[2], 32, [_A], shas[1]),
    ]


def test_a_false_good_step_is_caught_at_the_preceding_point(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One false result during the bisect moves the culprit one point later,
    and the run on the point before it finds the real break."""
    shas = env.repo(["", "", "a", "a", "a", "a"])
    points: dict[str, int | None] = {shas[i]: 30 + i for i in range(1, 6)}
    monkeypatch.setenv("FAKE_FLAKES", json.dumps(
        [{"sha": shas[2], "test": "test_a", "phase": "bisect", "as": "ok"}]))
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[5], _A, points=points)
    assert rc == 1
    assert result["rounds"][0]["culprit"] == shas[3]
    assert result["events"] == [nb.event_for_range(shas[0], shas[3], [_A], nb.REASON_UNVERIFIED)]


def test_a_false_bad_at_the_bisect_and_the_recheck_is_caught_at_the_next_point(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The commit the bisect visits first reads bad every time, though the
    break comes later. The next merge point passes, which a break cannot."""
    shas = env.repo(["", "", "", "", "", "", "a", "a"])
    points: dict[str, int | None] = {shas[i]: 30 + i for i in range(1, 8)}
    false_bad = _git(tmp_path / "repo", "rev-list", "--bisect", shas[7], f"^{shas[0]}").stdout.strip()
    assert shas.index(false_bad) < 5
    monkeypatch.setenv("FAKE_FLAKES", json.dumps(
        [{"sha": false_bad, "test": "test_a", "phase": p, "as": "bad"} for p in ("bisect", "check")]))
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=points)
    assert rc == 1
    assert result["rounds"][0]["culprit"] == false_bad
    assert result["events"] == [nb.event_for_range(shas[0], shas[7], [_A], nb.REASON_UNVERIFIED)]


def test_a_deadline_at_the_next_point_keeps_the_narrowed_range(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(_LINEAR)
    points: dict[str, int | None] = {shas[2]: 11, shas[5]: 12, shas[6]: 13, shas[7]: 14}
    monkeypatch.setenv("FAKE_ABORT_CHECK", shas[6])
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=points)
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[2], shas[5], [_A], nb.REASON_BUDGET)]
    assert env.stepped()[-1] == shas[6]


def test_following_point() -> None:
    order = ["c1", "c2", "c3"]
    assert nb.following_point(order, {"c1", "c3"}, "c1") == "c3"
    assert nb.following_point(order, None, "c1") == "c2"
    assert nb.following_point(order, None, "c3") is None


def test_a_file_renamed_after_the_break_is_not_blamed_on_the_rename(
    env: Env, tmp_path: Path, run_bisect: Runner,
) -> None:
    """The test broke at 1 under its old name and reads absent, so good,
    until the rename at 3."""
    shas = env.repo(["", "a", "a", "a", "a"], edits=(
        (0, "tests/test_old.py", "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n"),
        (3, "tests/test_old.py", None),
        (3, "tests/test_new.py", "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n"),
    ))
    new_id = "tests/test_new.py::test_a"
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[4], new_id,
                            points={shas[i]: 30 + i for i in range(1, 5)})
    assert rc == 1
    assert result["rounds"][0]["culprit"] == shas[3]
    assert result["events"] == [nb.event_for_range(shas[0], shas[3], [new_id], nb.REASON_RENAMED)]


def test_a_file_deleted_and_added_back_is_not_blamed_on_the_return(
    env: Env, tmp_path: Path, run_bisect: Runner,
) -> None:
    """The test broke at 2, was gone at 3-5, and came back at 6 still broken."""
    shas = env.repo(["", "", "a", "a", "a", "a", "a", "a"], edits=(
        (0, "tests/test_z.py", "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n"),
        (3, "tests/test_z.py", None),
        (6, "tests/test_z.py", "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n"),
    ))
    z_id = "tests/test_z.py::test_a"
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], z_id,
                            points={shas[i]: 30 + i for i in range(1, 8)})
    assert rc == 1
    assert result["rounds"][0]["culprit"] == shas[6]
    assert result["events"] == [nb.event_for_range(shas[0], shas[6], [z_id], nb.REASON_RETURNED)]


def test_a_file_copied_after_the_break_is_not_blamed_on_the_copy(
    env: Env, tmp_path: Path, run_bisect: Runner,
) -> None:
    """Git sees a copy when the source changes in the same commit."""
    body = "".join(f"def test_{name}():\n    pass\n\n\n" for name in ("a", "b", "c", "d"))
    shas = env.repo(["", "a", "a", "a", "a"], edits=(
        (0, "tests/test_old.py", body),
        (3, "tests/test_old.py", f"{body}def test_e():\n    pass\n"),
        (3, "tests/test_new.py", body),
    ))
    new_id = "tests/test_new.py::test_a"
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[4], new_id,
                            points={shas[i]: 30 + i for i in range(1, 5)})
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[3], [new_id], nb.REASON_RENAMED)]


def test_a_next_point_that_cannot_be_judged_keeps_the_blame(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """That commit says nothing about the culprit."""
    shas = env.repo(_LINEAR)
    points: dict[str, int | None] = {shas[2]: 11, shas[5]: 12, shas[6]: 13, shas[7]: 14}
    monkeypatch.setenv("FAKE_EXITS", json.dumps([{"sha": shas[6], "phase": "check", "rc": 125}]))
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=points)
    assert rc == 0
    assert result["events"] == [nb.event_for_culprit(shas[5], 12, [_A], shas[2])]
    assert result["rounds"][0]["verify"]["following_rc"] == 125


def test_a_test_function_removed_and_added_back_is_not_blamed_on_the_return(
    env: Env, tmp_path: Path, run_bisect: Runner,
) -> None:
    shas = env.repo(["", "A", "A", "A", "a", "a"])
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[5], _A,
                            points={shas[i]: 30 + i for i in range(1, 6)})
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[4], [_A], nb.REASON_RETURNED)]


@pytest.mark.parametrize(("rc", "reason"), [(125, nb.REASON_UNVERIFIED), (255, nb.REASON_BUDGET)])
def test_a_first_commit_that_cannot_be_judged_is_no_proof_a_test_is_new(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch, rc: int, reason: str,
) -> None:
    """The status left `returned` empty, so the test was blamed on the return."""
    shas = env.repo(["", "A", "A", "A", "a", "a"])
    monkeypatch.setenv("FAKE_EXITS", json.dumps([{"sha": shas[0], "phase": "check", "rc": rc}]))
    status, result = run_bisect(tmp_path / "repo", shas[0], shas[5], _A,
                                points={shas[i]: 30 + i for i in range(1, 6)})
    assert status == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[4], [_A], reason)]


def test_a_test_added_failing_is_blamed_on_the_merge_that_added_it(
    env: Env, tmp_path: Path, run_bisect: Runner,
) -> None:
    shas = env.repo(["", "", "", "a", "a"], edits=((3, "tests/test_new.py", "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n"),))
    new_id = "tests/test_new.py::test_a"
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[4], new_id,
                            points={shas[i]: 30 + i for i in range(1, 5)})
    assert rc == 0
    assert result["events"] == [nb.event_for_culprit(shas[3], 33, [new_id], shas[2])]


def test_a_repeated_false_good_at_the_real_culprit_stays_undetected(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documented in the module docstring: the same wrong result at the
    bisect and at the check moves the blame one merge point later."""
    shas = env.repo(["", "", "", "a", "a", "a", "a", "a"])
    monkeypatch.setenv("FAKE_FLAKES", json.dumps(
        [{"sha": shas[3], "test": "test_a", "phase": p, "as": "ok"} for p in ("bisect", "check")]))
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A,
                            points={shas[i]: 30 + i for i in range(1, 8)})
    assert rc == 0
    assert result["events"] == [nb.event_for_culprit(shas[4], 34, [_A], shas[3])]


def test_a_status_of_one_with_no_test_named_is_a_range(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(["", "", "ab", "ab"])
    monkeypatch.setenv("FAKE_NO_FAILED", "1")
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[3], _A, _B,
                            points={shas[i]: 30 + i for i in range(1, 4)})
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[3], [_A, _B], nb.REASON_UNVERIFIED)]


@pytest.mark.parametrize("report", ["missing", "corrupt"])
def test_a_missing_junit_report_names_no_pr(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch, report: str,
) -> None:
    """A test calling `os._exit(1)` at 1 blamed test b too, which breaks at 3."""
    shas = env.repo(["", "a", "a", "ab", "ab"])
    monkeypatch.setenv("FAKE_UV_JUNIT", report)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[4], _A, _B,
                            points={shas[i]: 30 + i for i in range(1, 5)}, step=False)
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[4], [_A, _B], nb.REASON_UNVERIFIED)]


def test_a_preceding_point_with_no_report_is_a_range(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a report there, the two tests cannot be told apart at the preceding point."""
    shas = env.repo(["", "", "ab", "ab"])
    monkeypatch.setenv("FAKE_EXITS", json.dumps([{"sha": shas[1], "phase": "check", "rc": 1}]))
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[3], _A, _B,
                            points={shas[i]: 30 + i for i in range(1, 4)})
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[3], [_A, _B], nb.REASON_UNVERIFIED)]


@pytest.mark.parametrize("rc", [125, 1], ids=["cannot-judge", "no-report"])
def test_a_preceding_point_that_cannot_be_judged_is_a_range(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch, rc: int,
) -> None:
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("FAKE_EXITS", json.dumps([{"sha": shas[2], "phase": "check", "rc": rc}]))
    status, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert status == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[7], [_A], nb.REASON_UNVERIFIED)]
    assert result["rounds"][0]["verify"]["preceding_rc"] == rc


def test_a_second_run_with_no_report_at_the_culprit_is_a_range(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("FAKE_EXITS", json.dumps([{"sha": shas[5], "phase": "check", "rc": 1, "nth": [2]}]))
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[7], [_A], nb.REASON_UNVERIFIED)]


@pytest.mark.parametrize("report", ["missing", "corrupt"])
def test_step_with_no_readable_report_is_bad_and_names_no_test(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, report: str,
) -> None:
    env.repo(["a"])
    _tests_file(tmp_path / "ids.txt", _A)
    failed = tmp_path / "failed.txt"
    monkeypatch.setenv("FAKE_UV_JUNIT", report)
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12", "--failed-out", str(failed)) == nb.STEP_BAD
    assert failed.read_text(encoding="utf-8") == ""


def test_tests_past_the_cap_keep_their_own_reason(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    """One range merged 50 `budget` tests and 5 `cap` tests, and the cap
    tests read as `budget`."""
    shas = env.repo(_LINEAR)
    ids = [f"tests/test_x.py::test_n{i}" for i in range(55)]
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], *ids, points=_linear_points(shas),
                            extra=("--budget-seconds", "0"))
    assert rc == 1
    expected = {t: nb.REASON_BUDGET for t in ids[:50]} | {t: nb.REASON_CAP for t in ids[50:]}
    assert [(e["key"], e["reasons"], sorted(e["tests"])) for e in result["events"]] == [
        (f"range-{shas[0][:12]}-{shas[7][:12]}", expected, sorted(ids)),
    ]


def test_reasons_use_the_ids_as_given(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(["", "a"])
    junit = "tests.test_x::test_a"
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[1], junit, points={shas[1]: 7},
                            extra=("--budget-seconds", "0"))
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[1], [junit], nb.REASON_BUDGET)]


def test_a_run_that_raises_keeps_what_it_found(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first round found its culprit; the second round raises."""
    shas = env.repo(["", "", "a", "a", "ab", "ab"])
    real = nb.Bisector.bisect_round

    def second_round_raises(self: nb.Bisector, lo: str, tests: list[str], n: int) -> Any:
        if n == 2:
            raise RuntimeError("git failed in the test")
        return real(self, lo, tests, n)

    monkeypatch.setattr(nb.Bisector, "bisect_round", second_round_raises)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[5], _A, _B,
                            points={shas[i]: 30 + i for i in range(1, 6)})
    assert rc == 2
    assert result["complete"] is False
    assert result["events"] == [
        nb.event_for_culprit(shas[2], 32, [_A], shas[1]),
        nb.event_for_range(shas[0], shas[5], [_B], nb.REASON_STOPPED),
    ]


@pytest.mark.parametrize(("rc", "reason"), [(125, nb.REASON_UNVERIFIED), (255, nb.REASON_BUDGET)])
def test_a_second_run_that_cannot_judge_the_culprit_leaves_no_test_out(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch, rc: int, reason: str,
) -> None:
    """Test b did not fail at the culprit, and got no event at all."""
    shas = env.repo(["", "a", "ab", "ab"])
    monkeypatch.setenv("FAKE_EXITS", json.dumps([{"sha": shas[1], "phase": "check", "rc": rc, "nth": [2]}]))
    status, result = run_bisect(tmp_path / "repo", shas[0], shas[3], _A, _B,
                                points={shas[i]: 30 + i for i in range(1, 4)})
    assert status == 1
    low, high = (shas[0], shas[3]) if rc == 125 else (shas[0], shas[1])
    assert result["events"] == [nb.event_for_range(low, high, [_A, _B], reason)]


def _exits(sha: str, rc: int, nth: int | None = None) -> str:
    entry: dict[str, Any] = {"sha": sha, "phase": "check", "rc": rc}
    return json.dumps([entry if nth is None else {**entry, "nth": [nth]}])


# What each case sets, by the commit indexes of `_EVERY_HISTORY`, where a breaks
# at 2, b at 4 and c never. Merge points are every commit.
_EVERY_HISTORY = ["", "", "a", "a", "ab", "ab", "ab", "ab"]
_EVERY_CASES: dict[str, dict[str, Any]] = {
    "two-culprits": {},
    "budget-gone": {"extra": ("--budget-seconds", "0")},
    "abort-in-the-bisect": {"env": {"FAKE_ABORT_AFTER": "1"}},
    "abort-at-every-check": {"env": {"FAKE_ABORT_CHECK": "1"}},
    "no-report": {"env": {"FAKE_NO_FAILED": "1"}},
    "culprit-skip": {"exit": (2, 125, None)},
    "preceding-skip": {"exit": (1, 125, None)},
    "second-run-skip": {"exit": (2, 125, 2)},
    "second-run-abort": {"exit": (2, 255, 2)},
    "next-point-skip": {"exit": (3, 125, None)},
    "next-point-abort": {"exit": (3, 255, None)},
    "raises": {"raises": True},
    "over-the-cap": {"extra_ids": [f"tests/test_x.py::test_n{i}" for i in range(55)]},
    "unmapped": {"extra_ids": ["tests/test_gone.py::test_a", "tests.test_gone::test_a"]},
}


@pytest.mark.parametrize("case", list(_EVERY_CASES))
def test_every_input_test_ends_in_exactly_one_event(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    spec = _EVERY_CASES[case]
    shas = env.repo(_EVERY_HISTORY)
    for name, value in spec.get("env", {}).items():
        monkeypatch.setenv(name, value)
    if "exit" in spec:
        at, status, nth = spec["exit"]
        monkeypatch.setenv("FAKE_EXITS", _exits(shas[at], status, nth))
    if spec.get("raises"):
        monkeypatch.setattr(nb.Bisector, "attribute", lambda *_: (_ for _ in ()).throw(RuntimeError("in the test")))
    ids = [_A, _B, _C, *spec.get("extra_ids", [])]
    _, result = run_bisect(tmp_path / "repo", shas[0], shas[7], *ids, extra=spec.get("extra", ()),
                           points={shas[i]: 30 + i for i in range(1, 8)})
    named = [t for event in result["events"] for t in event["tests"]]
    assert sorted(named) == sorted(ids), result["events"]


def test_an_earlier_result_is_replaced_before_any_work(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An old `complete: true` stayed during a new run and after it died."""
    shas = env.repo(["", "a"])
    _write(tmp_path / "result.json", {"complete": True, "good": "earlier", "events": []})

    _fail_git(monkeypatch, "worktree", "add")
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[1], _A, points={shas[1]: 7})
    assert rc == 2
    assert (result["complete"], result["good"]) == (False, shas[0])


def test_the_result_is_false_while_the_worktree_is_prepared(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(["", "a"])
    _write(tmp_path / "result.json", {"complete": True, "good": "earlier", "events": []})
    real = nb.Git.run
    seen: list[dict[str, Any]] = []

    def spy(self: nb.Git, *args: str, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("worktree", "add"):
            seen.append(json.loads((tmp_path / "result.json").read_text(encoding="utf-8")))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(nb.Git, "run", spy)
    assert run_bisect(tmp_path / "repo", shas[0], shas[1], _A, points={shas[1]: 7})[0] == 0
    assert (seen[0]["complete"], seen[0]["good"]) == (False, shas[0])


@pytest.mark.parametrize("target", ["missing-folder", "a-folder"])
def test_an_out_path_that_cannot_be_written_stops_before_any_work(
    env: Env, tmp_path: Path, target: str,
) -> None:
    shas = env.repo(["", "a"])
    out = tmp_path / "nowhere" / "r.json" if target == "missing-folder" else tmp_path
    rc = nb.main(["run", "--repo-root", str(tmp_path / "repo"), "--good", shas[0], "--bad", shas[1],
                  "--tests", str(_tests_file(tmp_path / "ids.txt", _A)), "--out", str(out),
                  "--step-cmd", env.step_cmd])
    assert rc == 2
    assert env.stepped() == []
    assert env.folders() == []
    if target == "a-folder":
        assert [p.name for p in tmp_path.parent.iterdir() if p.name.startswith(f".{tmp_path.name}.")] == []


def test_the_result_is_replaced_in_one_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reader polling the file during a run never sees half of it."""
    real = os.replace
    moves: list[tuple[Path, Path]] = []

    def replace(src: str, dst: Path) -> None:
        moves.append((Path(src), dst))
        real(src, dst)

    monkeypatch.setattr(nb.os, "replace", replace)
    out = tmp_path / "r.json"
    nb._write_json(out, {"a": 1})
    assert [(src.parent, dst) for src, dst in moves] == [(tmp_path, out)]
    assert json.loads(out.read_text(encoding="utf-8")) == {"a": 1}


def test_each_event_reaches_the_result_as_it_happens(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run killed outright keeps what it found, since no `finally` runs."""
    shas = env.repo(["", "", "a", "a", "ab", "ab"])
    real = nb.Bisector.bisect_round
    seen: list[dict[str, Any]] = []

    def spy(self: nb.Bisector, lo: str, tests: list[str], n: int) -> Any:
        if n == 2:
            seen.append(json.loads((tmp_path / "result.json").read_text(encoding="utf-8")))
        return real(self, lo, tests, n)

    monkeypatch.setattr(nb.Bisector, "bisect_round", spy)
    assert run_bisect(tmp_path / "repo", shas[0], shas[5], _A, _B, points={shas[i]: 30 + i for i in range(1, 6)})[0] == 0
    assert seen[0]["complete"] is False
    assert [e["key"] for e in seen[0]["events"]] == ["pr-32"]


def test_a_finished_run_is_marked_complete_and_leaves_no_temporary_file(
    env: Env, tmp_path: Path,
) -> None:
    shas = env.repo(["", "a"])
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = nb.main(["run", "--repo-root", str(tmp_path / "repo"), "--good", shas[0], "--bad", shas[1],
                  "--tests", str(_tests_file(tmp_path / "ids.txt", _A)), "--out", str(out_dir / "r.json"),
                  "--step-cmd", env.step_cmd])
    assert rc == 0
    assert [p.name for p in out_dir.iterdir()] == ["r.json"]
    assert json.loads((out_dir / "r.json").read_text(encoding="utf-8"))["complete"] is True


def test_tests_left_after_the_last_round_get_a_range(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(["", "a", "ab", "abc", "abcd", "abcd"])
    points = {shas[i]: 30 + i for i in range(1, 6)}
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[5], _A, _B, _C, _D, points=points)
    assert rc == 1
    assert result["events"] == [
        nb.event_for_culprit(shas[1], 31, [_A], shas[0]),
        nb.event_for_culprit(shas[2], 32, [_B], shas[1]),
        nb.event_for_culprit(shas[3], 33, [_C], shas[2]),
        nb.event_for_range(shas[3], shas[5], [_D], nb.REASON_ROUNDS),
    ]
    assert len(result["rounds"]) == nb.MAX_ROUNDS


def test_a_test_that_passes_at_bad_gets_a_range(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(["", "", "a"])
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[2], _A, _B, points={shas[1]: 31, shas[2]: 32})
    assert rc == 1
    assert result["events"] == [
        nb.event_for_culprit(shas[2], 32, [_A], shas[1]),
        nb.event_for_range(shas[2], shas[2], [_B], nb.REASON_PASSES_AT_BAD),
    ]


def test_bad_is_never_skipped_when_the_points_file_leaves_it_out(
    env: Env, tmp_path: Path, run_bisect: Runner,
) -> None:
    shas = env.repo(["", "", "", "", "", "", "", "a"])
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points={shas[2]: 11, shas[5]: 12})
    assert rc == 0
    assert result["events"] == [nb.event_for_culprit(shas[7], None, [_A], shas[5])]
    # Commits 1, 3, 4 and 6, and never bad itself.
    assert result["rounds"][0]["skipped"] == 4


def test_adjacent_commits_run_only_the_checks(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    """Git has nothing to bisect, and `git bisect run` would step `good`."""
    shas = env.repo(["", "a"])
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[1], _A, points={shas[1]: 7})
    assert rc == 0
    assert result["events"] == [nb.event_for_culprit(shas[1], 7, [_A], shas[0])]
    assert env.stepped() == [shas[1], shas[0], shas[1]]


def test_a_recheck_survives_a_dirty_tree(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("FAKE_DIRTY", "check")
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 0
    assert result["events"] == [nb.event_for_culprit(shas[5], 12, [_A], shas[2])]


def _repo_state(repo: Path) -> dict[str, str]:
    """What a run has to leave as it was in the repository it was pointed at."""
    return {command: _git(repo, *command.split()).stdout for command in (
        "rev-parse HEAD", "symbolic-ref HEAD", "status --porcelain", "ls-files -s", "for-each-ref",
        "worktree list --porcelain",
    )}


def _a_checkout_with_work(repo: Path) -> None:
    """A modified file and a staged one, which a checkout there would lose."""
    (repo / "tests" / "test_x.py").write_text("# edited\n", encoding="utf-8")
    (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    assert _git(repo, "add", "staged.txt").returncode == 0


def test_the_run_leaves_the_repository_as_it_found_it(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(["", "", "a", "a"])
    repo = tmp_path / "repo"
    _a_checkout_with_work(repo)
    before = _repo_state(repo)
    real = nb.Bisector.bisect
    worktrees: list[Path] = []

    def spy(self: nb.Bisector, good: str, tests: list[str]) -> None:
        worktrees.append(self.worktree)
        real(self, good, tests)

    monkeypatch.setattr(nb.Bisector, "bisect", spy)
    rc, result = run_bisect(repo, shas[0], shas[3], _A, points={shas[1]: 21, shas[2]: 22, shas[3]: 23})
    assert rc == 0
    assert result["rounds"][0]["log"], "the round kept no bisect log"
    assert _repo_state(repo) == before
    assert worktrees[0].parent.parent == env.tmp
    assert (repo / "tests" / "test_x.py").read_text(encoding="utf-8") == "# edited\n"


def test_two_runs_on_one_repository_each_get_their_own_folder(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(["", "a"])
    real = nb._remove_folder
    folders: list[Path] = []

    def spy(git: nb.Git, folder: Path) -> None:
        folders.append(folder)
        real(git, folder)

    monkeypatch.setattr(nb, "_remove_folder", spy)
    for _ in range(2):
        assert run_bisect(tmp_path / "repo", shas[0], shas[1], _A, points={shas[1]: 7})[0] == 0
    assert len(set(folders)) == 2
    assert _git(tmp_path / "repo", "worktree", "list").stdout.count("\n") == 1


def test_a_run_touches_no_other_folder_and_prunes_nothing(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    """`.bisect` was deleted even when tracked, and `git worktree prune` took
    the registration of a worktree someone moved."""
    shas = env.repo(["", "a"], edits=((0, ".bisect/tracked.txt", "tracked\n"),))
    repo, other = tmp_path / "repo", tmp_path / "other"
    (repo / ".nightly-bisect-worktree").write_text("not this script's\n", encoding="utf-8")
    assert _git(repo, "worktree", "add", "-q", "--detach", str(other), shas[0]).returncode == 0
    other.rename(tmp_path / "moved")
    before = _repo_state(repo)
    assert run_bisect(repo, shas[0], shas[1], _A, points={shas[1]: 7})[0] == 0
    assert _repo_state(repo) == before
    assert (repo / ".bisect" / "tracked.txt").is_file()
    assert (repo / ".nightly-bisect-worktree").is_file()
    assert (tmp_path / "moved" / "state.txt").is_file()


@pytest.mark.parametrize("phase", ["bisect", "check"])
def test_a_lost_git_file_cannot_reach_the_repository_above_it(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], phase: str,
) -> None:
    """`git checkout --force` walked up to the caller's repository and moved its
    checkout, losing a modified and a staged file. The temporary folder sits
    inside that repository here, so only the ceiling stops git."""
    shas = env.repo(_LINEAR)
    repo = tmp_path / "repo"
    _a_checkout_with_work(repo)
    before = _repo_state(repo)
    del before["worktree list --porcelain"]
    inside = repo / "tmp"
    inside.mkdir()
    env.tmp = inside
    monkeypatch.setattr(tempfile, "tempdir", str(inside))
    monkeypatch.setenv("FAKE_RM_GIT", phase)
    rc, result = run_bisect(repo, shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 2
    assert "not a git repository" in capsys.readouterr().err
    after = _repo_state(repo)
    del after["worktree list --porcelain"]
    assert after == before
    assert (repo / "tests" / "test_x.py").read_text(encoding="utf-8") == "# edited\n"
    assert [(e["kind"], e["reasons"]) for e in result["events"]] == [("range", {_A: nb.REASON_STOPPED})]


def test_the_step_and_its_children_cannot_find_a_repository_above_the_worktree(
    env: Env, tmp_path: Path, run_bisect: Runner,
) -> None:
    """Fake step for the bisect and the checks, then the real step's `uv` children."""
    shas = env.repo(["", "", "", "a", "a"])
    points: dict[str, int | None] = {shas[i]: 20 + i for i in range(1, 5)}
    for step in (True, False):
        env.ceiling_log.unlink(missing_ok=True)
        assert run_bisect(tmp_path / "repo", shas[0], shas[4], _A, points=points, step=step)[0] == 0
        rows = [row.rsplit(" ", 1) for row in env.ceiling_log.read_text(encoding="utf-8").splitlines()]
        kinds = {"step bisect", "step check"} if step else {"uv sync", "uv collect", "uv run"}
        assert {kind for kind, _ in rows} == kinds
        assert len({ceiling for _, ceiling in rows}) == 1
        assert Path(rows[0][1]).parent == env.tmp
        assert Path(rows[0][1]).name.startswith("nightly-bisect-")


def test_a_worktree_git_cannot_remove_goes_by_its_path(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shas = env.repo(["", "a"])
    _fail_git(monkeypatch, "worktree", "remove")
    assert run_bisect(tmp_path / "repo", shas[0], shas[1], _A, points={shas[1]: 7})[0] == 0
    assert "::warning::removing" in capsys.readouterr().err
    assert env.folders() == []


def test_a_read_only_folder_goes_and_a_link_is_not_followed(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read-only folder stops `git worktree remove`; `chmod -R` runs next,
    and what a symbolic link names is not this run's to change."""
    shas = env.repo(["", "a"])
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep\n", encoding="utf-8")
    victim.chmod(0o500)
    monkeypatch.setenv("FAKE_READONLY", str(victim))
    try:
        assert run_bisect(tmp_path / "repo", shas[0], shas[1], _A, points={shas[1]: 7})[0] == 0
        assert (victim / "keep.txt").is_file()
        assert stat.S_IMODE(victim.stat().st_mode) == 0o500
    finally:
        victim.chmod(0o700)


def test_a_folder_that_cannot_be_removed_is_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(nb.shutil, "rmtree", lambda *_, **__: None)
    nb._remove_folder(nb.Git(tmp_path), tmp_path)
    assert f"::warning::{tmp_path} could not be removed" in capsys.readouterr().err


def test_a_signal_cannot_stop_the_removal(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(["", "a"])
    real = nb._remove_folder
    during: list[Any] = []

    def spy(git: nb.Git, folder: Path) -> None:
        during.append((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)))
        real(git, folder)

    monkeypatch.setattr(nb, "_remove_folder", spy)
    assert run_bisect(tmp_path / "repo", shas[0], shas[1], _A, points={shas[1]: 7})[0] == 0
    assert during == [(signal.SIG_IGN, signal.SIG_IGN)]


def test_a_dirty_tree_does_not_stop_the_reset(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The step dirties a file that differs between commits: the bisect
    cannot go on, and a plain `git bisect reset` refuses too."""
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("FAKE_DIRTY", "bisect")
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 1
    assert result["reset_errors"] == []


def _fail_git(monkeypatch: pytest.MonkeyPatch, *prefix: str) -> None:
    """Make every `git` call of the script that starts with `prefix` fail."""
    real = nb.Git.run

    def run(self: nb.Git, *args: str, check: bool = True, timeout: float = 120) -> subprocess.CompletedProcess[str]:
        if args[: len(prefix)] != prefix:
            return real(self, *args, check=check, timeout=timeout)
        if check:
            raise RuntimeError(f"git {' '.join(args)} failed in the test")
        return subprocess.CompletedProcess(["git", *args], 1, "", "refused by the test")

    monkeypatch.setattr(nb.Git, "run", run)


def test_a_reset_that_fails_is_reported(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shas = env.repo(_LINEAR)
    _fail_git(monkeypatch, "bisect", "reset")
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert result["reset_errors"] == ["refused by the test"]
    assert "::warning::git bisect reset failed: refused by the test" in capsys.readouterr().err
    assert rc == 0


def test_a_reset_that_times_out_is_reported_and_the_result_kept(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(_LINEAR)
    real = nb.Git.run

    def run(self: nb.Git, *args: str, check: bool = True, timeout: float = 120) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("bisect", "reset"):
            raise subprocess.TimeoutExpired(["git", *args], timeout)
        return real(self, *args, check=check, timeout=timeout)

    monkeypatch.setattr(nb.Git, "run", run)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 0
    assert len(result["reset_errors"]) == 1
    assert "timed out" in result["reset_errors"][0]
    assert result["complete"] is True


def test_the_folder_goes_when_the_round_raises(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(_LINEAR)
    _fail_git(monkeypatch, "rev-parse", "--verify", "refs/bisect/bad")
    rc, _ = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 2
    assert _git(tmp_path / "repo", "worktree", "list").stdout.count("\n") == 1


def test_a_relative_repo_root_works(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative root exited 2 and left `repo/repo/` behind."""
    shas = env.repo(["", "a"])
    monkeypatch.chdir(tmp_path)
    ids = _tests_file(tmp_path / "ids.txt", _A)
    rc = nb.main(["run", "--repo-root", "repo", "--good", shas[0], "--bad", shas[1], "--tests", str(ids),
                  "--out", "result.json", "--step-cmd", env.step_cmd])
    assert rc == 0
    assert not (tmp_path / "repo" / "repo").exists()
    points = nb.main(["merge-points", "--repo-root", "repo", "--good", shas[0], "--bad", shas[1],
                      "--out", "points.json", "--dry-run"])
    assert points == 0


def test_python_and_extras_reach_the_step(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(["", "a"])
    rc, _ = run_bisect(tmp_path / "repo", shas[0], shas[1], _A, points={shas[1]: 7}, step=False,
                       extra=("--python", "3.13", "--extra", "archive", "--extra", "onboard-llm"))
    assert rc == 0
    assert env.uv_calls()[0] == ["sync", "--frozen", "--extra", "archive", "--extra", "onboard-llm", "--python", "3.13"]


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
    assert result["events"] == [nb.event_for_culprit(shas[3], 42, [module_id], shas[1])]


@pytest.mark.parametrize("target", ["::tests.sub", "tests/sub/"])
def test_a_folder_whose_conftest_stops_importing_is_bisected(
    env: Env, tmp_path: Path, run_bisect: Runner, target: str,
) -> None:
    """`tests/sub/` read as unmapped, and pytest's status 4 for it as a skip."""
    shas = env.repo(["", "", "", ""], edits=(
        (0, "tests/sub/conftest.py", ""),
        (2, "tests/sub/conftest.py", "import BROKEN\n"),
    ))
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[3], target,
                            points={shas[1]: 21, shas[2]: 22, shas[3]: 23}, step=False)
    assert rc == 0, result
    assert result["events"] == [nb.event_for_culprit(shas[2], 22, [target], shas[1])]


def test_a_non_ascii_test_file_maps(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    """`ls-tree` quotes the name without `-z`, and the id read as unmapped."""
    shas = env.repo(["", "", "a"], edits=((0, "tests/test_caf\u00e9.py", "def test_a():\n    pass\n"),))
    node_id = "tests/test_caf\u00e9.py::test_a"
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[2], node_id, points={shas[1]: 1, shas[2]: 2})
    assert rc == 0, result
    assert result["events"] == [nb.event_for_culprit(shas[2], 2, [node_id], shas[1])]


def test_a_non_ascii_file_renamed_after_the_break_is_a_range(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    body = "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n"
    shas = env.repo(["", "a", "a", "a", "a"], edits=(
        (0, "tests/test_old.py", body), (3, "tests/test_old.py", None), (3, "tests/test_caf\u00e9.py", body),
    ))
    new_id = "tests/test_caf\u00e9.py::test_a"
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[4], new_id, points={shas[i]: 30 + i for i in range(1, 5)})
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[3], [new_id], nb.REASON_RENAMED)]


def test_a_junit_id_is_bisected_like_its_node_id(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    """`test_outcome_record.py` names a test `tests.test_x::test_a`. Read as
    a path, that names no file, and the step called every commit good."""
    shas = env.repo(["", "", "a", "a"])
    junit = "tests.test_x::test_a"
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[3], junit,
                            points={shas[1]: 21, shas[2]: 22, shas[3]: 23}, step=False)
    assert rc == 0, result
    assert result["events"] == [nb.event_for_culprit(shas[2], 22, [junit], shas[1])]


def test_ids_that_name_no_file_get_a_range_without_a_step(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(["", "", "a", "a"])
    unread = ["tests.test_gone::test_a", "::tests.test-gone", "tests/test_gone.py::test_a"]
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[3], *unread,
                            points={shas[1]: 21, shas[2]: 22, shas[3]: 23})
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[0], shas[3], unread, nb.REASON_UNMAPPED)]
    assert env.stepped() == []


def test_a_hyphenated_module_id_is_judged_by_its_file(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(["", "", "", ""], edits=(
        (0, "tests/test-y.py", "def test_c():\n    pass\n"),
        (2, "tests/test-y.py", "import BROKEN\n\n\ndef test_c():\n    pass\n"),
    ))
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[3], "::tests.test-y",
                            points={shas[1]: 21, shas[2]: 22, shas[3]: 23}, step=False)
    assert rc == 0, result
    assert [(e["key"], e["tests"]) for e in result["events"]] == [("pr-22", ["::tests.test-y"])]


def test_the_step_gets_git_dir_and_its_children_do_not(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    """Under `GIT_DIR`, `git init` in a test's temporary directory re-creates
    the worktree's git directory, and the suite's tests do that."""
    shas = env.repo(["", "", "", "a", "a"])
    points: dict[str, int | None] = {shas[i]: 20 + i for i in range(1, 5)}
    assert run_bisect(tmp_path / "repo", shas[0], shas[4], _A, points=points)[0] == 0
    seen = env.env_log.read_text(encoding="utf-8").splitlines()
    assert any(row.startswith("step bisect ") and not row.endswith(" None") for row in seen), seen
    env.env_log.unlink()
    shutil.rmtree(tmp_path / "repo")
    env.repo(["", "", "", "a", "a"])
    assert run_bisect(tmp_path / "repo", shas[0], shas[4], _A, points=points, step=False)[0] == 0
    seen = env.env_log.read_text(encoding="utf-8").splitlines()
    assert any(row.startswith("uv ") for row in seen)
    assert [row for row in seen if not row.endswith(" None")] == []


def test_the_children_lose_every_variable_git_ties_to_a_repository(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    listed = _git(tmp_path, "rev-parse", "--local-env-vars").stdout.split()
    assert "GIT_DIR" in listed
    for var in {*listed, "GIT_PREFIX"}:
        monkeypatch.setenv(var, "x")
    assert [var for var in nb.repo_free_env() if var in {*listed, "GIT_PREFIX"}] == []


def test_an_outside_git_dir_does_not_redirect_the_script(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "nowhere"))
    out = tmp_path / "points.json"
    assert _merge_points(tmp_path / "repo", shas[0], shas[7], out, "--dry-run") == 0


def test_dry_run_changes_nothing(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(_LINEAR)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas),
                            extra=("--dry-run",))
    assert (rc, result) == (0, {})
    assert env.stepped() == []
    assert env.folders() == []
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


# ------------------------------------------------- the canary's triage

# Input and output of a real run of `nightly_canary.py regressions`, from
# ci/issue-1786-nightly-canary-script at 67ef5347, before that script reached
# this branch. `--repo-root` held tests/test_x.py, tests/test_y.py and tests/sub/.
_TRIAGE_FIXTURE = Path(__file__).parent / "fixtures" / "nightly_triage_1786.json"
_CANARY_SCRIPT = _REPO / "scripts" / "nightly_canary.py"


def _triage_case() -> dict[str, Any]:
    return json.loads(_TRIAGE_FIXTURE.read_text(encoding="utf-8"))


def test_the_bisect_reads_the_canarys_real_triage(env: Env, tmp_path: Path) -> None:
    """Test a breaks at 2, tests/test_y.py and the conftest.py of tests/sub/
    stop importing at 3, test b breaks at 4, and tests/test_gone.py has no
    file. Each entry of the triage takes its own path through the real step,
    under the fake `uv`."""
    shas = env.repo(["", "", "a", "am", "abm", "abm"], edits=(
        (0, "tests/sub/conftest.py", ""),
        (3, "tests/sub/conftest.py", "import BROKEN\n"),
    ))
    triage = _write(tmp_path / "triage.json", _triage_case()["triage"])
    points = _write(tmp_path / "points.json", {shas[i]: 30 + i for i in range(1, 6)})
    out = tmp_path / "result.json"
    rc = nb.main(["run", "--repo-root", str(tmp_path / "repo"), "--good", shas[0], "--bad", shas[5],
                  "--triage", str(triage), "--points", str(points), "--out", str(out)])
    result = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 1, result
    assert result["events"] == [
        nb.event_for_culprit(shas[2], 32, ["tests.test_x::test_a"], shas[1]),
        nb.event_for_culprit(shas[3], 33, ["::tests.sub", "::tests.test_y"], shas[2]),
        nb.event_for_culprit(shas[4], 34, ["tests.test_x::test_b"], shas[3]),
        nb.event_for_range(shas[0], shas[5], ["tests.test_gone::test_a"], nb.REASON_UNMAPPED),
    ]


def test_the_triage_fixture_is_what_the_canary_writes(tmp_path: Path) -> None:
    """Fails when `regressions` changes the shape this script reads. Holds
    nothing until `nightly_canary.py` reaches the same branch; then a change
    to either script has to update the fixture here."""
    if not _CANARY_SCRIPT.is_file():
        return
    case = _triage_case()
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    for name in ("test_x.py", "test_y.py"):
        (repo / "tests" / name).write_text("", encoding="utf-8")
    (repo / "tests" / "sub").mkdir()
    branch = tmp_path / "branch"
    run_dir = branch / case["pointer"]["canary"]["path"]
    run_dir.mkdir(parents=True)
    _write(run_dir / "record.json", case["last_green"])
    _write(branch / "last-green.json", case["pointer"])
    current = _write(tmp_path / "current.json", case["current"])
    out = tmp_path / "triage.json"
    r = subprocess.run(
        [sys.executable, str(_CANARY_SCRIPT), "regressions", "--current", str(current),
         "--last-green-dir", str(branch), "--canary-legs", "3.12,3.13", "--repo-root", str(repo), "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=False,
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == case["triage"]


@pytest.mark.parametrize(
    "change",
    [
        {"format": 2},
        {"candidates": [{"test": "tests.test_x::test_a", "node": "tests/test_x.py::test_a"}]},
        {"collection_errors": [{"test": "::tests.test_y", "path": "tests/test_y.py"}]},
        {"new_failing": None},
        {"format": True},
        {"format": 1.0},
        {"format": "1"},
        {"candidates": "abc"},
        {"candidates": {}},
        {"candidates": [None]},
        {"candidates": [{"test": None, "node_id": _A}]},
        {"candidates": [{"test": "", "node_id": _A}]},
        {"candidates": [{"test": 5, "node_id": _A}]},
        {"candidates": [{"test": "x", "node_id": [_A]}]},
        {"candidates": [{"test": "x", "node_id": "state.txt"}]},
        {"new_failing": [{"test": "x", "node_id": "tests/test_x.py"}]},
        {"new_failing": [{"test": "x", "node_id": "tests/test_x.py::"}]},
        {"collection_errors": [{"test": "x", "module": "tests.test_y"}]},
        {"collection_errors": [{"test": "x", "module": "state.txt"}]},
    ],
    ids=["format", "node-id-renamed", "module-renamed", "not-a-list", "format-true", "format-float", "format-text",
         "text-list", "object-list", "null-row", "null-test", "empty-test", "number-test", "list-node-id",
         "tracked-non-test-file", "node-id-no-name", "node-id-empty-name", "module-dotted", "module-non-test-file"],
)
def test_a_triage_of_another_shape_exits_2(env: Env, tmp_path: Path, change: dict[str, Any]) -> None:
    shas = env.repo(["", "a"])
    triage = _write(tmp_path / "triage.json", {**_triage_case()["triage"], **change})
    rc = nb.main(["run", "--repo-root", str(tmp_path / "repo"), "--good", shas[0], "--bad", shas[1],
                  "--triage", str(triage), "--out", str(tmp_path / "r.json"), "--step-cmd", env.step_cmd])
    assert rc == 2
    assert env.stepped() == []


@pytest.mark.parametrize("key", ["candidates", "new_failing", "collection_errors", "format"])
def test_a_triage_missing_a_key_exits_2(env: Env, tmp_path: Path, key: str) -> None:
    shas = env.repo(["", "a"])
    case = _triage_case()["triage"]
    del case[key]
    rc = nb.main(["run", "--repo-root", str(tmp_path / "repo"), "--good", shas[0], "--bad", shas[1],
                  "--triage", str(_write(tmp_path / "triage.json", case)), "--out", str(tmp_path / "r.json"),
                  "--step-cmd", env.step_cmd])
    assert rc == 2
    assert env.stepped() == []


def test_a_valid_triage_with_no_test_exits_0_with_an_empty_result(env: Env, tmp_path: Path) -> None:
    shas = env.repo(["", "a"])
    triage = _write(tmp_path / "triage.json", {"format": 1, "candidates": [], "new_failing": [], "collection_errors": []})
    out = tmp_path / "r.json"
    rc = nb.main(["run", "--repo-root", str(tmp_path / "repo"), "--good", shas[0], "--bad", shas[1],
                  "--triage", str(triage), "--out", str(out), "--step-cmd", env.step_cmd])
    assert rc == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert (result["events"], result["tests"], result["complete"]) == ([], [], True)
    assert env.stepped() == []
    assert env.folders() == []


def test_tests_given_a_triage_exits_2(env: Env, tmp_path: Path) -> None:
    """Each line of the JSON read as a test id."""
    shas = env.repo(["", "a"])
    triage = _write(tmp_path / "triage.json", _triage_case()["triage"])
    rc = nb.main(["run", "--repo-root", str(tmp_path / "repo"), "--good", shas[0], "--bad", shas[1],
                  "--tests", str(triage), "--out", str(tmp_path / "r.json"), "--step-cmd", env.step_cmd])
    assert rc == 2
    assert not (tmp_path / "r.json").exists()


def test_a_triage_naming_a_file_missing_at_bad_gets_a_range() -> None:
    triage: dict[str, Any] = {
        "format": 1, "new_failing": [],
        "candidates": [
            {"test": "tests.gone::test_a", "node_id": "tests/gone.py::test_a"},
            {"test": "tests.test_x::test_a", "node_id": _A},
            {"test": "tests.test_x::test_a", "node_id": _A},
        ],
        "collection_errors": [{"test": "::tests.test_y", "module": "tests/test_y.py"},
                              {"test": "::tests.sub", "module": "tests/sub/"},
                              {"test": "::tests.gone_sub", "module": "tests/gone_sub/"},
                              {"test": "::tests.test_z", "module": None}],
    }
    mapped, unmapped = nb.map_triage(triage, nb.tree_paths({"tests/test_x.py", "tests/test_y.py", "tests/sub/a.py"}))
    assert mapped == {_A: ["tests.test_x::test_a"], "tests/test_y.py": ["::tests.test_y"],
                      "tests/sub/": ["::tests.sub"]}
    assert unmapped == ["tests.gone::test_a", "::tests.gone_sub", "::tests.test_z"]


def test_run_takes_tests_or_a_triage_and_not_both(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["run", "--good", "a", "--bad", "b", "--out", str(tmp_path / "r.json")]
    for extra in ([], ["--tests", "x", "--triage", "y"]):
        with pytest.raises(SystemExit) as exit_info:
            nb.main([*base, *extra])
        assert exit_info.value.code == 2
    assert "--tests" in capsys.readouterr().err


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


@pytest.mark.parametrize("prs", ["empty", "elsewhere"])
def test_merge_points_with_no_merge_in_the_range_means_every_commit(
    env: Env, tmp_path: Path, run_bisect: Runner, capsys: pytest.CaptureFixture[str], prs: str,
) -> None:
    """A successful, empty answer used to leave `bad` alone as a point, and
    the run blamed it with nothing tested between it and good."""
    shas = env.repo(_LINEAR)
    _write(env.gh_out, [] if prs == "empty" else [{"number": 9, "mergeCommit": {"oid": shas[0]}}])
    out = tmp_path / "points.json"
    assert _merge_points(tmp_path / "repo", shas[0], shas[7], out) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {}
    assert "::warning::" in capsys.readouterr().err
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points={})
    assert rc == 0
    assert result["events"][0]["key"] == f"commit-{shas[4][:12]}"


def test_run_given_points_with_no_pr_in_the_range_tests_every_commit(
    env: Env, tmp_path: Path, run_bisect: Runner, capsys: pytest.CaptureFixture[str],
) -> None:
    shas = env.repo(_LINEAR)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points={shas[7]: None, "f" * 40: 5})
    assert rc == 0
    assert result["events"][0]["key"] == f"commit-{shas[4][:12]}"
    assert "::warning::" in capsys.readouterr().err


@pytest.mark.parametrize(("count", "kept"), [(500, False), (499, True)])
def test_merge_points_at_the_list_limit_means_every_commit(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str], count: int, kept: bool,
) -> None:
    """`gh` returned exactly its limit with 509 merges in range, and the
    points it left out moved the culprit to the wrong PR."""
    shas = env.repo(_LINEAR)
    prs = [{"number": 1000 + i, "mergeCommit": {"oid": f"{i:040x}"}} for i in range(count - 1)]
    _write(env.gh_out, [{"number": 12, "mergeCommit": {"oid": shas[5]}}, *prs])
    out = tmp_path / "points.json"
    assert _merge_points(tmp_path / "repo", shas[0], shas[7], out) == 0
    points = json.loads(out.read_text(encoding="utf-8"))
    assert points == ({shas[5]: 12, shas[7]: None} if kept else {})
    assert ("::warning::" in capsys.readouterr().err) is not kept


@pytest.mark.parametrize("merge", [None, {}, {"oid": ""}, "abc"], ids=["null", "no-oid", "empty-oid", "not-an-object"])
def test_a_pr_with_no_merge_commit_means_every_commit(
    env: Env, tmp_path: Path, capsys: pytest.CaptureFixture[str], merge: Any,
) -> None:
    """The real culprit's PR had a null `mergeCommit`, and one other PR in
    range was enough to accept the list, so the blame moved to the next PR."""
    shas = env.repo(_LINEAR)
    _write(env.gh_out, [{"number": 12, "mergeCommit": merge},
                        {"number": 13, "mergeCommit": {"oid": shas[7]}}])
    out = tmp_path / "points.json"
    assert _merge_points(tmp_path / "repo", shas[0], shas[7], out) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {}
    assert "PR 12 has no merge commit" in capsys.readouterr().err


def test_merge_points_ignores_output_from_a_failed_gh(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    shas = env.repo(_LINEAR)
    _write(env.gh_out, [{"number": 12, "mergeCommit": {"oid": shas[5]}}])
    monkeypatch.setenv("FAKE_GH_FAIL", "with-output")
    out = tmp_path / "points.json"
    assert _merge_points(tmp_path / "repo", shas[0], shas[7], out) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {}
    assert "HTTP 502" in capsys.readouterr().err


def test_merge_points_leaves_out_a_merged_side_branch(env: Env, tmp_path: Path) -> None:
    """A side branch from before `good` is in `good..bad` but not on the
    ancestry path, so a bisect could not call it good or bad."""
    shas = env.repo(["", "", "", ""])
    repo = tmp_path / "repo"
    for args in (["checkout", "-q", "-b", "side", shas[0]], ["commit", "-q", "--allow-empty", "-m", "side"]):
        assert _git(repo, *args).returncode == 0
    side = _git(repo, "rev-parse", "HEAD").stdout.strip()
    for args in (["checkout", "-q", "main"], ["merge", "-q", "--no-ff", "--no-edit", "side"]):
        r = _git(repo, *args)
        assert r.returncode == 0, r.stderr
    merge = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _write(env.gh_out, [
        {"number": 5, "mergeCommit": {"oid": side}},
        {"number": 6, "mergeCommit": {"oid": merge}},
        {"number": 7, "mergeCommit": {"oid": shas[2]}},
    ])
    out = tmp_path / "points.json"
    assert _merge_points(repo, shas[1], merge, out) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {shas[2]: 7, merge: 6}


def test_a_gh_that_hangs_means_every_commit(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    shas = env.repo(_LINEAR)
    real = subprocess.run
    timeouts: list[float] = []

    def run(argv: list[str], **kwargs: Any) -> Any:
        if argv[0] != "gh":
            return real(argv, **kwargs)
        timeouts.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(nb.subprocess, "run", run)
    out = tmp_path / "points.json"
    assert _merge_points(tmp_path / "repo", shas[0], shas[7], out) == 0
    assert timeouts == [nb._API_TIMEOUT_SECONDS]
    assert json.loads(out.read_text(encoding="utf-8")) == {}
    assert "::warning::" in capsys.readouterr().err


def test_gh_runs_in_the_repo_root_without_git_variables(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without `--repo`, gh reads the remote of the folder it runs in."""
    shas = env.repo(_LINEAR)
    _write(env.gh_out, [])
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "nowhere"))
    assert _merge_points(tmp_path / "repo", shas[0], shas[7], tmp_path / "p.json") == 0
    _, folder, git_dir = env.env_log.read_text(encoding="utf-8").split()
    assert Path(folder).resolve() == (tmp_path / "repo").resolve()
    assert git_dir == "None"


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
        ({"past_deadline": False, "module_rcs": [4]}, nb.STEP_BAD),
        ({"past_deadline": False, "module_rcs": [3]}, nb.STEP_SKIP),
        ({"past_deadline": False, "module_rcs": [5, 4, 3]}, nb.STEP_SKIP),
        ({"past_deadline": True, "module_rcs": [2]}, nb.STEP_ABORT),
        ({"past_deadline": False, "sync_rc": 1, "module_rcs": [2]}, nb.STEP_SKIP),
        ({"past_deadline": False, "unmapped": True}, nb.STEP_SKIP),
        ({"past_deadline": True, "unmapped": True}, nb.STEP_ABORT),
    ],
    ids=["deadline", "deadline-first", "sync-fails", "sync-timeout", "collect-error", "collect-timeout",
         "nothing-collected", "no-id-present", "run-passes", "run-fails", "run-interrupted",
         "run-usage", "run-timeout", "module-error", "module-error-wins", "module-collects",
         "module-collects-run-fails", "module-timeout", "target-conftest", "target-internal-error",
         "target-any-other-status", "module-deadline", "module-sync",
         "unmapped", "unmapped-deadline"],
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


@pytest.mark.parametrize("target", ["tests/sub/", "::tests.sub"])
def test_step_with_an_absent_folder_is_good(env: Env, tmp_path: Path, target: str) -> None:
    env.repo([""])
    _tests_file(tmp_path / "ids.txt", target)
    absent = tmp_path / "absent.txt"
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12", "--absent-out", str(absent)) == nb.STEP_GOOD
    assert absent.read_text(encoding="utf-8") == f"{target}\n"
    assert [c[0] for c in env.uv_calls()] == ["sync"]


def test_step_with_an_absent_module_file_is_good(env: Env, tmp_path: Path) -> None:
    env.repo(["m"])
    _tests_file(tmp_path / "ids.txt", "tests.test_gone")
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12") == nb.STEP_GOOD
    assert [c[0] for c in env.uv_calls()] == ["sync"]


def test_step_given_a_junit_id_maps_it_to_the_checked_out_tree(env: Env, tmp_path: Path) -> None:
    env.repo(["a"])
    _tests_file(tmp_path / "ids.txt", "tests.test_x::test_a")
    failed = tmp_path / "failed.txt"
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12", "--failed-out", str(failed)) == nb.STEP_BAD
    assert failed.read_text(encoding="utf-8") == "tests.test_x::test_a\n"
    assert env.uv_calls()[-1][-1] == _A


@pytest.mark.parametrize("unread", ["tests.test_gone::test_a", "tests..test_x::test_a", "::a::b"])
def test_step_given_an_id_with_no_file_skips(env: Env, tmp_path: Path, unread: str) -> None:
    env.repo(["a"])
    _tests_file(tmp_path / "ids.txt", unread)
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12") == nb.STEP_SKIP
    assert env.uv_calls() == []


@pytest.mark.parametrize("stage", ["sync", "collect", "run"])
def test_step_timeout_ends_the_whole_child_group(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    """A timeout used to kill only the direct child, and pytest under uv lived on."""
    env.repo(["a"])
    _tests_file(tmp_path / "ids.txt", _A)
    monkeypatch.setenv("FAKE_UV_HANG", stage)
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12", "--step-timeout", "1") == nb.STEP_SKIP
    assert _survivors(env.pids()) == []


def test_a_terminated_step_ends_its_children(env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`run` signals the step's group; the step's children have their own."""
    env.repo(["a"])
    _tests_file(tmp_path / "ids.txt", _A)
    monkeypatch.setenv("FAKE_UV_HANG", "sync")
    argv = [*nb.default_step_cmd(), "--tests", str(tmp_path / "ids.txt"), "--deadline", "9e12", "--step-timeout", "60"]
    rc, _ = nb.run_group(argv, 2, cwd=tmp_path / "repo")
    assert rc is None
    assert _survivors(env.pids()) == []


def test_run_group_is_not_held_by_a_process_that_left_the_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid.txt"
    code = (
        "import pathlib, subprocess, sys;"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'], start_new_session=True);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid)); print('done')"
    )
    # Waiting on a pipe the escaped child holds would hit this timeout: status None.
    rc, output = nb.run_group([sys.executable, "-c", code], 15)
    os.kill(int(pid_file.read_text(encoding="utf-8")), 9)
    assert (rc, output.strip()) == (0, "done")


def test_run_group_kills_what_a_finished_child_left_in_its_group(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid.txt"
    code = (
        "import pathlib, subprocess, sys;"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)']);"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(p.pid))"
    )
    assert nb.run_group([sys.executable, "-c", code], 15)[0] == 0
    assert _survivors([int(pid_file.read_text(encoding="utf-8"))]) == []


def test_run_group_reports_a_missing_program() -> None:
    assert nb.run_group(["/nonexistent/program"], 5) == (127, "")


def test_step_names_the_absent_tests(env: Env, tmp_path: Path) -> None:
    env.repo(["a"])
    gone_file, gone_test = "tests/test_gone.py::test_a", "tests/test_x.py::test_gone"
    _tests_file(tmp_path / "ids.txt", _A, gone_file, gone_test, "tests.test_gone", "tests.test_y")
    absent = tmp_path / "absent.txt"
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12", "--absent-out", str(absent)) == nb.STEP_BAD
    assert sorted(absent.read_text(encoding="utf-8").split()) == sorted([gone_file, gone_test, "tests.test_gone"])


def test_step_that_cannot_read_its_input_aborts(env: Env, tmp_path: Path) -> None:
    env.repo(["a"])
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12") == nb.STEP_ABORT


@pytest.mark.parametrize(
    "flags",
    [("--deadline", "abc"), ("--deadline", "nan"), ("--deadline", "inf"), ("--deadline", "9e12", "--step-timeout", "-5"),
     ("--deadline", "9e12", "--step-timeout", "1" + "0" * 400), ("--deadline", "9e12", "--step-timeout", "1" + "0" * 5000),
     ("--deadline", "9e12", "--no-such-flag")],
    ids=["text", "nan", "inf", "negative", "400-digits", "too-long-for-int", "unknown"],
)
def test_step_with_a_bad_flag_skips(env: Env, tmp_path: Path, flags: tuple[str, ...]) -> None:
    """`git bisect run` reads a usage error's 2, and a traceback's 1, as a bad commit."""
    env.repo(["a"])
    _tests_file(tmp_path / "ids.txt", _A)
    assert _step(tmp_path, tmp_path / "repo", *flags) == nb.STEP_SKIP
    assert env.uv_calls() == []


def test_step_given_no_flags_at_all_skips(tmp_path: Path) -> None:
    assert nb.main(["step"]) == nb.STEP_SKIP


@pytest.mark.parametrize(
    ("error", "status"),
    [(OverflowError("in the test"), nb.STEP_SKIP), (ZeroDivisionError("in the test"), nb.STEP_SKIP),
     (subprocess.TimeoutExpired("x", 1), nb.STEP_ABORT), (OSError("in the test"), nb.STEP_ABORT)],
    ids=["overflow", "any-other-bug", "timeout", "unreadable"],
)
def test_step_errors_never_exit_as_a_bad_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, status: int,
) -> None:
    def raises(_: Any) -> int:
        raise error

    monkeypatch.setattr(nb, "cmd_step", raises)
    assert nb.main(["step", "--tests", "ids.txt", "--deadline", "9e12"]) == status


def test_run_turns_an_unexpected_error_into_a_status_of_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A traceback's status of 1 means "a test got a range"."""
    def raises(_: Any) -> int:
        raise OverflowError("in the test")

    monkeypatch.setattr(nb, "cmd_run", raises)
    assert nb.main(["run", "--good", "a", "--bad", "b", "--tests", "t", "--out", "o"]) == 2


@pytest.mark.parametrize("flag", ["--budget-seconds", "--step-timeout"])
@pytest.mark.parametrize("value", ["-5", "1" + "0" * 400, "x"], ids=["negative", "400-digits", "text"])
def test_run_refuses_a_bad_number(tmp_path: Path, capsys: pytest.CaptureFixture[str], flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as exit_info:
        nb.main(["run", "--good", "a", "--bad", "b", "--tests", "t", "--out", "o", flag, value])
    assert exit_info.value.code == 2
    assert flag in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", str(7 * 24 * 3600)])
def test_seconds_takes_its_whole_range(value: str) -> None:
    assert nb._seconds(value) == int(value)


def test_step_dry_run_runs_nothing(env: Env, tmp_path: Path) -> None:
    env.repo(["a"])
    _tests_file(tmp_path / "ids.txt", _A)
    assert _step(tmp_path, tmp_path / "repo", "--deadline", "9e12", "--dry-run") == 0
    assert env.uv_calls() == []


# --------------------------------------------------------------- signals

# Starts `run`, sends it a signal once the fake has named a process, and
# prints its status. A driver, so that the bounded `subprocess.run` around it
# gives the test its exit.
_SIGNALLER = '''\
import subprocess, sys, time
from pathlib import Path
signum, pids, *argv = sys.argv[1:]
proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
limit = time.monotonic() + 60
while not (Path(pids).is_file() and Path(pids).read_text(encoding="utf-8").strip()):
    if time.monotonic() > limit:
        proc.kill()
        sys.exit("the fake never started")
    time.sleep(0.05)
proc.send_signal(int(signum))
print(proc.wait(timeout=60))
'''


def _signal_a_run(env: Env, tmp_path: Path, shas: list[str], signum: int, *extra: str) -> int:
    """The status of a `run` that got `signum` while the hanging fake was up."""
    r = subprocess.run(
        [sys.executable, "-c", _SIGNALLER, str(signum), str(env.pids_file),
         sys.executable, str(_REPO / "scripts" / "nightly_bisect.py"), "run", "--repo-root", str(tmp_path / "repo"),
         "--good", shas[0], "--bad", shas[-1], "--out", str(tmp_path / "result.json"), *extra],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=110, check=False,
    )
    assert r.returncode == 0, r.stderr
    return int(r.stdout)


def _wait_for(condition: Callable[[], Any], what: str) -> None:
    limit = time.monotonic() + 60
    while not condition():
        assert time.monotonic() < limit, f"{what} never happened"
        time.sleep(0.05)


@pytest.mark.parametrize("signum", [15, 2], ids=["SIGTERM", "SIGINT"])
def test_a_signal_ends_the_children_removes_the_folder_and_saves(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, signum: int,
) -> None:
    """SIGTERM left `git bisect run` and the step alive."""
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("FAKE_HANG", "30")
    ids = str(_tests_file(tmp_path / "ids.txt", _A, _B))
    assert _signal_a_run(env, tmp_path, shas, signum, "--tests", ids, "--step-cmd", env.step_cmd) == 128 + signum
    assert _survivors(env.pids()) == []
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["complete"] is False
    assert sorted(t for e in result["events"] for t in e["tests"]) == [_A, _B]
    assert env.folders() == []
    assert _git(tmp_path / "repo", "worktree", "list").stdout.count("\n") == 1


def test_a_signal_runs_the_step_handler_before_the_kill(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real step's children have their own group, and only its handler ends them."""
    shas = env.repo(["", "", "a"])
    monkeypatch.setenv("FAKE_UV_HANG", "sync")
    ids = str(_tests_file(tmp_path / "ids.txt", _A))
    assert _signal_a_run(env, tmp_path, shas, 15, "--tests", ids) == 128 + 15
    assert _survivors(env.pids()) == []


def test_a_killed_run_keeps_the_next_one_out_until_its_children_end(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An orphan wrote into the next run's bisect: `was both good and bad`."""
    shas = env.repo(_LINEAR)
    monkeypatch.setenv("FAKE_HANG", "30")
    ids = str(_tests_file(tmp_path / "ids.txt", _A, _B))
    assert _signal_a_run(env, tmp_path, shas, 9, "--tests", ids, "--step-cmd", env.step_cmd) == -9
    orphan = env.pids()[0]
    group = os.getpgid(orphan)
    assert _running(orphan)
    monkeypatch.delenv("FAKE_HANG")
    rc, _ = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 2
    assert len(env.stepped()) == 1, "the second run tested a commit"
    os.killpg(group, 9)
    _wait_for(lambda: not _running(orphan), "the orphan's end")

    def lock_is_free() -> bool:
        return run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))[0] == 0

    _wait_for(lock_is_free, "the lock's release")


def test_the_lock_and_the_handlers_are_back_after_a_run(env: Env, tmp_path: Path, run_bisect: Runner) -> None:
    shas = env.repo(["", "a"])

    def mine(*_: Any) -> None:
        raise AssertionError("no signal is sent here")

    previous = signal.signal(signal.SIGTERM, mine), signal.signal(signal.SIGINT, mine)
    try:
        for _ in range(2):
            assert run_bisect(tmp_path / "repo", shas[0], shas[1], _A, points={shas[1]: 7})[0] == 0
        assert nb._HELD_FDS == []
        assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == (mine, mine)
    finally:
        signal.signal(signal.SIGTERM, previous[0])
        signal.signal(signal.SIGINT, previous[1])


def test_a_second_signal_cannot_stop_the_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    try:
        with pytest.raises(nb.Interrupted):
            nb._interrupt(signal.SIGTERM, None)
        assert (signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)) == (signal.SIG_IGN, signal.SIG_IGN)
    finally:
        signal.signal(signal.SIGTERM, previous[0])
        signal.signal(signal.SIGINT, previous[1])


# ---------------------------------------------------------------- budget

class _FakeClock:
    """The script's `time` module, with a clock a test moves."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def time(self) -> float:
        return self.now

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def _hanging_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hangs: str, *extra: str) -> tuple[int, int, float]:
    """The real step in-process on 50 module ids. A child whose arguments
    hold `hangs` takes its whole timeout on the fake clock. Returns the step's
    status, how many children it started, and the seconds that passed."""
    (tmp_path / "tests").mkdir()
    ids = [f"tests/mod{i}.py" for i in range(50)]
    for path in ids:
        (tmp_path / path).write_text("", encoding="utf-8")
    clock = _FakeClock()
    started = clock.now
    calls: list[list[str]] = []

    def run_group(argv: list[str], timeout: float, **_: Any) -> tuple[int | None, str]:
        calls.append(argv)
        if hangs in argv:
            clock.now += timeout
            return None, ""
        return 0, ""

    monkeypatch.setattr(nb, "run_group", run_group)
    monkeypatch.setattr(nb, "time", clock)
    monkeypatch.chdir(tmp_path)
    rc = nb.main(["step", "--tests", str(_tests_file(tmp_path / "ids.txt", *ids)), *extra])
    return rc, len(calls), clock.now - started


@pytest.mark.parametrize("hangs", ["sync", "--collect-only"])
def test_step_timeout_bounds_the_whole_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hangs: str,
) -> None:
    """50 module ids with a hung collect took 156.8 s under a budget of 5:
    each one got at least 1 s and 2 s of grace, past the step's own end."""
    rc, calls, spent = _hanging_step(tmp_path, monkeypatch, hangs, "--deadline", "9e12", "--step-timeout", "3")
    assert rc == nb.STEP_SKIP
    assert spent == 3
    assert calls == (1 if hangs == "sync" else 2)


def test_the_deadline_bounds_the_step_when_it_comes_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rc, calls, spent = _hanging_step(tmp_path, monkeypatch, "sync", "--deadline", "1002", "--step-timeout", "300")
    assert rc == nb.STEP_SKIP
    assert (calls, spent) == (1, 2)


def test_the_run_gives_its_children_only_the_budget_and_the_margin(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound on `git bisect run` was the budget plus a whole step."""
    shas = env.repo(_LINEAR)
    real = nb.run_group
    timeouts: dict[str, list[float]] = {"bisect": [], "check": []}

    def spy(argv: list[str], timeout: float, **kwargs: Any) -> tuple[int | None, str]:
        timeouts["bisect" if argv[0] == "git" else "check"].append(timeout)
        return real(argv, timeout, **kwargs)

    monkeypatch.setattr(nb, "run_group", spy)
    rc, _ = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas),
                       extra=("--budget-seconds", "100", "--step-timeout", "90"))
    assert rc == 0
    assert timeouts["bisect"] and max(timeouts["bisect"]) <= 100 + 15
    assert timeouts["check"] and max(timeouts["check"]) <= 90 + 15


def test_a_check_past_the_deadline_runs_nothing(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(_LINEAR)
    real = nb.Bisector.bisect_round

    def then_out_of_time(self: nb.Bisector, lo: str, tests: list[str], n: int) -> Any:
        out = real(self, lo, tests, n)
        self.deadline = 0.0
        return out

    monkeypatch.setattr(nb.Bisector, "bisect_round", then_out_of_time)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 1
    assert result["events"] == [nb.event_for_range(shas[2], shas[5], [_A], nb.REASON_BUDGET)]
    assert "check" not in env.step_log.read_text(encoding="utf-8")


def test_the_skip_loop_stops_at_the_deadline(
    env: Env, tmp_path: Path, run_bisect: Runner, monkeypatch: pytest.MonkeyPatch,
) -> None:
    shas = env.repo(_LINEAR)
    clock = _FakeClock()
    real = nb.Git.run
    skips: list[tuple[str, ...]] = []

    def run(self: nb.Git, *args: str, check: bool = True, timeout: float = 120) -> subprocess.CompletedProcess[str]:
        if args[:2] == ("bisect", "skip"):
            skips.append(args)
            clock.now = 1e10
        return real(self, *args, check=check, timeout=timeout)

    monkeypatch.setattr(nb.Git, "run", run)
    monkeypatch.setattr(nb, "time", clock)
    monkeypatch.setattr(nb, "_SKIP_CHUNK", 1)
    rc, result = run_bisect(tmp_path / "repo", shas[0], shas[7], _A, points=_linear_points(shas))
    assert rc == 1
    assert len(skips) == 1
    assert [set(e["reasons"].values()) for e in result["events"]] == [{nb.REASON_BUDGET}]


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
        ("::tests.test-y", "tests/test-y.py"),
        ("tests/sub/", "tests/sub/"),
        ("../sub/", None),
        ("tests..test_x", None),
        ("/etc/test_x.py", None),
        ("tests/../../test_x.py", None),
        ("tests/test x", None),
    ],
)
def test_module_file(test_id: str, path: str | None) -> None:
    assert nb.module_file(test_id) == path


@pytest.mark.parametrize(
    ("test_id", "canonical"),
    [
        ("tests/test_x.py::test_a", "tests/test_x.py::test_a"),
        ("tests/test_gone.py::test_a", "tests/test_gone.py::test_a"),
        ("tests.test_x::test_a", "tests/test_x.py::test_a"),
        ("tests.test_x.TestC::test_a[p.q]", "tests/test_x.py::TestC::test_a[p.q]"),
        ("tests.test_x.TestC.TestD::test_a", "tests/test_x.py::TestC::TestD::test_a"),
        ("tests.sub.test_z::test_a[a::b]", "tests/sub/test_z.py::test_a[a::b]"),
        ("tests.test_x", "tests/test_x.py"),
        ("::tests.test_x", "tests/test_x.py"),
        ("::tests.sub", "tests/sub/"),
        ("tests/sub/", "tests/sub/"),
        ("::tests.nope", "tests/nope.py"),
        ("tests.test_gone::test_a", None),
        ("tests.test-y::test_a", None),
        ("tests.twice.test_z::test_a", None),
        ("tests.test_x.Not-A-Class::test_a", None),
        ("tests.test_x[p]::test_a", None),
        ("tests.test_x::", None),
        ("tests..test_x::test_a", None),
        ("tests/test_x::test_a", None),
        ("/etc/test_x.py::test_a", None),
        ("tests/../test_x.py::test_a", None),
        ("::", None),
    ],
)
def test_canonical_id(test_id: str, canonical: str | None) -> None:
    files = nb.tree_paths({"tests/test_x.py", "tests/sub/test_z.py", "tests/twice.py", "tests/twice/test_z.py"})
    assert nb.canonical_id(test_id, files.__contains__) == canonical


def test_tree_paths_names_each_folder_with_a_slash() -> None:
    assert nb.tree_paths(["a/b/c.py", "d.py"]) == {"a/b/c.py", "d.py", "a/", "a/b/"}


def test_map_ids_keeps_each_given_form() -> None:
    mapped, unmapped = nb.map_ids(
        ["tests.test_x::test_a", _A, "tests/test_gone.py::test_a", "tests.test_gone", "tests.test_gone::test_a"],
        {"tests/test_x.py"},
    )
    assert mapped == {_A: ["tests.test_x::test_a", _A]}
    assert unmapped == ["tests/test_gone.py::test_a", "tests.test_gone", "tests.test_gone::test_a"]


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


def test_select_points() -> None:
    prs: list[dict[str, Any]] = [{"number": 1, "mergeCommit": {"oid": "c1"}}]
    assert nb.select_points(["c1", "c2"], prs, "c2") == {"c1": 1, "c2": None}
    assert nb.select_points(["c1", "c2"], [], "c2") == {}
    assert nb.select_points(["c2"], prs, "c2") == {}


def test_a_pr_that_only_merged_bad_is_a_merge_point() -> None:
    assert nb.select_points(["c1", "c2"], [{"number": 1, "mergeCommit": {"oid": "c2"}}], "c2") == {"c2": 1}
    assert nb.usable_points({"c2": 1}, ["c1", "c2"]) == {"c2": 1}
    assert nb.usable_points({"c2": None}, ["c1", "c2"]) == {}


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
    assert [(e["key"], e["tests"], e["reasons"]) for e in events] == [
        (f"range-{'a' * 12}-{'b' * 12}", ["t1", "t2"], {"t1": nb.REASON_ROUNDS, "t2": nb.REASON_CAP}),
    ]


def test_parse_tests_and_points() -> None:
    assert nb.parse_tests("# c\n\n a::b \na::b\nc::d\n") == ["a::b", "c::d"]
    assert nb.parse_points({"s": 1, "t": None}) == {"s": 1, "t": None}
    for bad in ([], {"s": "1"}, {"s": True}):
        with pytest.raises(ValueError):
            nb.parse_points(bad)
