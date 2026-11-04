"""#1786 shape B: the nightly canary's record, triage input and results branch.

`scripts/nightly_canary.py` maps a leg's red tests to node ids for the
rerun, sorts them into flaky and confirmed, builds the night's format-1
record, compares the canary legs with the last green night and writes the
night to a checkout of the `test-results` branch. These tests run its pure
functions on synthetic JUnit reports in the shapes pytest 9 writes
(classname and name of a module test, a nested class, parameter ids with
dots, brackets, spaces and `::`, and a collection error with an empty
classname), and run its subcommands against temporary files and a temporary
git repository. Two tests run real pytest over a small project, so the
synthetic shapes cannot drift from what pytest writes without one of them
failing. Nothing here calls the network or `gh`.
"""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "nightly_canary.py"
sys.path.insert(0, str(_REPO / "scripts"))

import nightly_canary as nc  # noqa: E402
import test_outcome_record as tor  # noqa: E402

pytestmark = pytest.mark.timeout(120)

SHA1 = "1" * 40
SHA2 = "2" * 40
CREATED = "2026-11-04T02:17:00Z"


# ----------------------------------------------------------- fixtures

def _case(classname: str, name: str, body: str = "") -> str:
    name = name.replace('"', "&quot;")
    if body:
        return f'<testcase classname="{classname}" name="{name}" time="0.001">{body}</testcase>'
    return f'<testcase classname="{classname}" name="{name}" time="0.001" />'


def _report(*cases: str) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
        f'<testsuite name="pytest" tests="{len(cases)}">{"".join(cases)}</testsuite></testsuites>'
    )


_FAIL = '<failure message="AssertionError: boom">assert 1 == 2</failure>'
_ERROR = '<error message="failed on setup with RuntimeError">RuntimeError</error>'
_COLLECT_ERROR = '<error message="collection failure">ModuleNotFoundError</error>'
_XPASS_STRICT = '<failure message="[XPASS(strict)] known bug">[XPASS(strict)] known bug</failure>'
_SKIP = '<skipped type="pytest.skip" message="needs sqlite 3.45">t.py:3: needs sqlite 3.45</skipped>'


def _junit(outcomes: dict[str, str]) -> str:
    """A report holding `classname::name -> outcome` for the outcomes used here."""
    bodies = {tor.PASSED: "", tor.FAILED: _FAIL, tor.ERROR: _ERROR, tor.SKIPPED: _SKIP, tor.XPASSED: _XPASS_STRICT}
    cases = []
    for test, outcome in outcomes.items():
        classname, _, name = test.partition("::")
        cases.append(_case(classname, name, bodies[outcome]))
    return _report(*cases)


def _tree(root: Path, *files: str) -> Path:
    for rel in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("", encoding="utf-8")
    return root


def _bundle(
    directory: Path,
    outcomes: dict[str, str] | None,
    rc: int | None = None,
    rerun: dict[str, str] | None = None,
    env: dict[str, Any] | None = None,
) -> Path:
    """A leg's bundle as the suite job leaves it."""
    directory.mkdir(parents=True, exist_ok=True)
    if outcomes is not None:
        (directory / "junit.xml").write_text(_junit(outcomes), encoding="utf-8")
        if rc is None:
            rc = 1 if nc.red_ids(outcomes) else 0
        verdict = nc.classify_leg(rc, outcomes, rerun)
        (directory / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
        if rerun is not None:
            (directory / "rerun.xml").write_text(_junit(rerun), encoding="utf-8")
    (directory / "env.json").write_text(json.dumps(env or {"python": "3.x"}), encoding="utf-8")
    return directory


def _night(
    sha: str,
    legs: dict[str, tuple[dict[str, str] | None, dict[str, str] | None]],
    roles: dict[str, str] | None = None,
    run_id: int = 100,
    run_attempt: int = 1,
    created_at: str = CREATED,
    rcs: dict[str, int] | None = None,
) -> dict[str, Any]:
    """A night's record: leg -> (full-run outcomes or None, rerun outcomes).
    A leg's pytest rc is 1 when it has a red test and 0 otherwise, unless
    `rcs` names it."""
    junit: dict[str, str | None] = {}
    verdicts: dict[str, dict[str, Any] | None] = {}
    for leg, (outcomes, rerun) in legs.items():
        if outcomes is None:
            junit[leg], verdicts[leg] = None, None
            continue
        junit[leg] = _junit(outcomes)
        rc = (rcs or {}).get(leg, 1 if nc.red_ids(outcomes) else 0)
        verdicts[leg] = nc.classify_leg(rc, outcomes, rerun)
    all_roles = {leg: nc.CANARY for leg in legs} | (roles or {})
    return nc.build_nightly_record(
        junit, verdicts, {leg: None for leg in legs}, all_roles, {leg: None for leg in legs},
        sha, run_id, run_attempt, "schedule", created_at,
    )


_A = "tests.test_a::test_one"
_B = "tests.test_a::test_two"
_C = "tests.test_b::test_three"


# ------------------------------------------------------------- failed-ids

@pytest.mark.parametrize(
    ("test", "expected"),
    [
        ("tests.test_a::test_x", "tests/test_a.py::test_x"),
        ("tests.test_a.TestGroup::test_x", "tests/test_a.py::TestGroup::test_x"),
        ("tests.test_a.TestOuter.TestInner::test_in", "tests/test_a.py::TestOuter::TestInner::test_in"),
        ("tests.sub.deep.test_p::test_param[a.b]", "tests/sub/deep/test_p.py::test_param[a.b]"),
        ("tests.sub.deep.test_p::test_param[x y]", "tests/sub/deep/test_p.py::test_param[x y]"),
        ("tests.sub.deep.test_p::test_param[[q]]", "tests/sub/deep/test_p.py::test_param[[q]]"),
        ("tests.sub.deep.test_p::test_param[a::b]", "tests/sub/deep/test_p.py::test_param[a::b]"),
        ("tests.sub.deep.test_p::test_param[\\xe9]", "tests/sub/deep/test_p.py::test_param[\\xe9]"),
        ("tests.foo.TestF::test_z", "tests/foo.py::TestF::test_z"),
        ("tests.pkg.test_c::test_y", "tests/pkg/test_c.py::test_y"),
        ("tests.foo.test_bar::test_y", None),
        ("tests.dep[x].test_br::test_y", None),
        ("tests.dep[x.test_open::test_y", None),
        ("tests.dep]x.test_close::test_y", None),
        ("tests.test_a::", None),
        ("test_top::test_y", "test_top.py::test_y"),
        ("tests.foo.2.test_d::test_y", None),
        ("::tests.test_broken", None),
        ("tests.gone::test_x", None),
        ("tests.test_a", None),
    ],
    ids=["module", "class", "nested-class", "param-dot", "param-space", "param-brackets",
         "param-colons", "param-escape", "module-beside-dir", "module-in-dir",
         "two-prefixes-are-files", "bracket-in-path", "open-bracket-in-path", "close-bracket-in-path", "no-test-name",
         "module-at-the-root", "dotted-dir-over-a-file",
         "collection-error", "deleted-module", "no-name"],
)
def test_a_junit_id_maps_to_the_node_id_pytest_reruns(tmp_path: Path, test: str, expected: str | None) -> None:
    """Each shape pytest 9 writes, and the shapes pytest would refuse."""
    root = _tree(tmp_path, "tests/test_a.py", "tests/sub/deep/test_p.py", "tests/foo.py", "tests/foo/test_bar.py",
                 "tests/pkg/test_c.py", "tests/dep[x]/test_br.py", "tests/dep[x/test_open.py", "tests/dep]x/test_close.py",
                 "test_top.py")
    assert nc.node_id(test, lambda rel: (root / rel).is_file()) == expected


def test_a_collection_error_never_maps_even_beside_a_file_named_dot_py(tmp_path: Path) -> None:
    root = _tree(tmp_path, ".py")
    assert nc.node_id("::tests.test_broken", lambda rel: (root / rel).is_file()) is None


def test_failed_ids_writes_every_red_node_id_and_the_outputs(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo", "tests/test_a.py", "tests/test_b.py")
    junit = tmp_path / "junit.xml"
    junit.write_text(_report(
        _case("tests.test_a", "test_pass"),
        _case("tests.test_a", "test_fail", _FAIL),
        _case("tests.test_a.TestG", "test_err", _ERROR),
        _case("tests.test_b", "test_xpass[p.1]", _XPASS_STRICT),
        _case("tests.test_b", "test_skip", _SKIP),
        _case("", "tests.test_gone", _COLLECT_ERROR),
    ), encoding="utf-8")
    out, gh_out = tmp_path / "ids.txt", tmp_path / "gh_output"
    gh_out.write_text("earlier=1\n", encoding="utf-8")
    rc = nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(root), "--out", str(out),
                  "--github-output", str(gh_out)])
    assert rc == 0
    assert out.read_text(encoding="utf-8").splitlines() == [
        "tests/test_a.py::TestG::test_err",
        "tests/test_a.py::test_fail",
        "tests/test_b.py::test_xpass[p.1]",
    ]
    assert gh_out.read_text(encoding="utf-8") == "earlier=1\ncount=4\nrerun=true\nmass_failure=false\n"


@pytest.mark.parametrize(("n", "rerun", "exit_status"), [(50, "true", 0), (51, "false", 1)])
def test_more_than_fifty_red_tests_is_a_mass_failure_with_no_rerun(
    tmp_path: Path, n: int, rerun: str, exit_status: int,
) -> None:
    root = _tree(tmp_path / "repo", "tests/test_a.py")
    junit = tmp_path / "junit.xml"
    junit.write_text(_report(*(_case("tests.test_a", f"test_{i}", _FAIL) for i in range(n))), encoding="utf-8")
    out, gh_out = tmp_path / "ids.txt", tmp_path / "gh_output"
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(root), "--out", str(out),
                    "--github-output", str(gh_out)]) == exit_status
    if exit_status == 0:
        assert len(out.read_text(encoding="utf-8").splitlines()) == n
    else:
        assert not out.exists()
    assert f"rerun={rerun}\n" in gh_out.read_text(encoding="utf-8")
    assert f"mass_failure={'true' if n > 50 else 'false'}\n" in gh_out.read_text(encoding="utf-8")


def test_a_run_whose_only_red_test_maps_to_nothing_is_not_rerun(tmp_path: Path) -> None:
    """An empty `@file` would make pytest run the whole suite."""
    plan = nc.plan_rerun({"::tests.test_gone": tor.ERROR}, lambda rel: False, 50)
    assert plan == {"count": 1, "rerun": False, "mass_failure": False, "ids": [], "unmapped": ["::tests.test_gone"]}


@pytest.mark.parametrize(
    "cases",
    [
        [_case("tests.test_a", f"test_{i}", _FAIL) for i in range(51)],
        [_case("", "tests.test_gone", _COLLECT_ERROR)],
        [_case("tests.test_a", "test_pass")],
    ],
    ids=["mass-failure", "nothing-maps", "nothing-red"],
)
def test_with_nothing_to_rerun_no_file_exists_and_the_exit_status_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cases: list[str],
) -> None:
    """pytest given an empty `@file` runs the whole suite; given none it
    exits 4. A stale file from an earlier step goes too."""
    root = _tree(tmp_path / "repo", "tests/test_a.py")
    junit = tmp_path / "junit.xml"
    junit.write_text(_report(*cases), encoding="utf-8")
    out = tmp_path / "ids.txt"
    out.write_text("tests/test_a.py::stale\n", encoding="utf-8")
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(root), "--out", str(out)]) == 1
    assert not out.exists()
    assert "rerun=false" in capsys.readouterr().out


def test_with_something_to_rerun_the_decision_prints_without_a_github_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    root = _tree(tmp_path / "repo", "tests/test_a.py")
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit({"tests.test_a::test_x": tor.FAILED}), encoding="utf-8")
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(root),
                    "--out", str(tmp_path / "ids.txt")]) == 0
    assert "rerun=true" in capsys.readouterr().out


def test_a_dry_run_keeps_the_exit_status_and_a_stale_file(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo", "tests/test_a.py")
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit({"tests.test_a::test_x": tor.PASSED}), encoding="utf-8")
    out = tmp_path / "ids.txt"
    out.write_text("stale\n", encoding="utf-8")
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(root), "--out", str(out),
                    "--dry-run"]) == 1
    assert out.read_text(encoding="utf-8") == "stale\n"


def test_failed_ids_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _tree(tmp_path / "repo", "tests/test_a.py")
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit({"tests.test_a::test_x": tor.FAILED}), encoding="utf-8")
    out, gh_out = tmp_path / "ids.txt", tmp_path / "gh_output"
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(root), "--out", str(out),
                    "--github-output", str(gh_out), "--dry-run"]) == 0
    assert not out.exists() and not gh_out.exists()
    printed = capsys.readouterr().out
    assert "tests/test_a.py::test_x" in printed and "rerun=true" in printed


def test_failed_ids_needs_a_readable_report_and_a_repo_root(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    out = str(tmp_path / "ids.txt")
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(tmp_path), "--out", out]) == 2
    junit.write_text("<testsuites><testcase", encoding="utf-8")
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(tmp_path), "--out", out]) == 2
    junit.write_text(_junit({_A: tor.PASSED}), encoding="utf-8")
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(tmp_path / "no"), "--out", out]) == 2


_LONG_ID = "a." * 3000 + "b::test_x"
_LONG_COLLECTION_ID = "::" + "a." * 3000 + "b"


def test_a_classname_too_long_for_a_path_stays_confirmed_and_the_rest_still_reruns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    root = _tree(tmp_path / "repo", "tests/test_a.py")
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit({_A: tor.FAILED, _LONG_ID: tor.FAILED}), encoding="utf-8")
    out = tmp_path / "ids.txt"
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(root), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").splitlines() == ["tests/test_a.py::test_one"]
    assert _LONG_ID in capsys.readouterr().err
    assert nc.classify_leg(1, {_A: tor.FAILED, _LONG_ID: tor.FAILED}, {_A: tor.PASSED})["confirmed"] == [_LONG_ID]


def test_regressions_survives_ids_too_long_for_a_path(tmp_path: Path) -> None:
    tonight = _night(SHA2, {"3.12": ({_LONG_ID: tor.FAILED, _LONG_COLLECTION_ID: tor.ERROR}, None),
                            "3.13": ({_A: tor.PASSED}, None)})
    current = tmp_path / "record.json"
    current.write_text(json.dumps(tonight), encoding="utf-8")
    (tmp_path / "branch").mkdir()
    (tmp_path / "repo").mkdir()
    out = tmp_path / "t.json"
    assert nc.main(["regressions", "--current", str(current), "--last-green-dir", str(tmp_path / "branch"),
                    "--repo-root", str(tmp_path / "repo"), "--out", str(out)]) == 0
    triage = json.loads(out.read_text(encoding="utf-8"))
    assert triage["collection_errors"][0]["module"] is None
    assert {f["test"]: f["node_id"] for f in triage["events"][0]["failures"]}[_LONG_ID] is None


@pytest.mark.parametrize("command", ["failed-ids", "verdict"])
@pytest.mark.parametrize(("value", "status"), [("-1", 2), ("0", 1)], ids=["negative", "zero"])
def test_max_refuses_a_negative_value(tmp_path: Path, command: str, value: str, status: int) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit({_A: tor.FAILED}), encoding="utf-8")
    args = {"failed-ids": ["--repo-root", str(tmp_path), "--out", str(tmp_path / "ids.txt")], "verdict": ["--rc", "1"]}
    try:
        code = nc.main([command, "--junit", str(junit), "--max", value, *args[command]])
    except SystemExit as exc:
        code = exc.code
    assert code == status


def _pytest(project: Path, *args: str, fail: bool = False) -> subprocess.CompletedProcess[str]:
    """Real pytest in a throwaway project. Tests read FAIL from the environment."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_") and k != "FAIL"}
    if fail:
        env["FAIL"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *args],
        cwd=project, env=env, capture_output=True, text=True, timeout=90,
    )


_FAILS_ON_FLAG = "import os\nBAD = os.environ.get('FAIL') == '1'\n"


def test_ids_that_would_not_round_trip_stay_out_and_the_rerun_does_not_exit_4(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Real pytest: a class sharing a name with a same-stem file, a directory
    holding `[x]` and one holding a dot. Before this rule one of them in the
    file made pytest exit 4, so every flaky test in the rerun read as confirmed."""
    project = tmp_path / "project"
    files = {
        "pyproject.toml": "[tool.pytest.ini_options]\ntestpaths = ['tests']\n",
        "tests/test_col.py": _FAILS_ON_FLAG + "class TestX:\n    def test_y(self):\n        assert not BAD\n",
        "tests/test_col/TestX.py": "",
        "tests/test_dep[x]/test_br.py": _FAILS_ON_FLAG + "def test_br():\n    assert not BAD\n",
        "tests/v1.2/test_dot.py": _FAILS_ON_FLAG + "def test_dot():\n    assert not BAD\n",
        "tests/test_p.py": _FAILS_ON_FLAG + (
            "import pytest\n"
            "@pytest.mark.parametrize('v', ['a.b', 'x y', '[q]', 'c::d', '--opt', '@file', '#c'])\n"
            "def test_v(v):\n    assert not BAD\n"
        ),
    }
    for rel, body in files.items():
        (project / rel).parent.mkdir(parents=True, exist_ok=True)
        (project / rel).write_text(body, encoding="utf-8")
    full = _pytest(project, "--junitxml=junit.xml", fail=True)
    assert full.returncode == 1, full.stdout
    out = tmp_path / "ids.txt"
    assert nc.main(["failed-ids", "--junit", str(project / "junit.xml"), "--repo-root", str(project),
                    "--out", str(out)]) == 0
    unmappable = {"tests.test_col.TestX::test_y", "tests.test_dep[x].test_br::test_br", "tests.v1.2.test_dot::test_dot"}
    notices = capsys.readouterr().err
    assert all(test in notices for test in unmappable)
    assert len(out.read_text(encoding="utf-8").splitlines()) == 7
    rerun = _pytest(project, f"@{out}", "--junitxml=rerun.xml")
    assert rerun.returncode == 0, rerun.stdout + rerun.stderr
    verdict = nc.classify_leg(
        1,
        tor.parse_junit((project / "junit.xml").read_text(encoding="utf-8")),
        tor.parse_junit((project / "rerun.xml").read_text(encoding="utf-8")),
    )
    assert set(verdict["confirmed"]) == unmappable
    assert len(verdict["flaky"]) == 7


_REAL_MODULE = _FAILS_ON_FLAG + (
    "import pytest\n"
    "def test_pass():\n    pass\n"
    "def test_always_fails():\n    assert False\n"
    "def test_flaky():\n    assert not BAD\n"
    "def test_skip():\n    pytest.skip('not here')\n"
    "@pytest.mark.xfail(reason='known')\n"
    "def test_xfail():\n    assert False\n"
    "@pytest.mark.parametrize('v', ['a.b', 'c d'])\n"
    "def test_param(v):\n    assert not BAD\n"
    "class TestGroup:\n    def test_in_class(self):\n        assert False\n"
)


def test_real_pytest_output_runs_through_failed_ids_and_verdict(tmp_path: Path) -> None:
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']\n", encoding="utf-8")
    (project / "tests" / "test_m.py").write_text(_REAL_MODULE, encoding="utf-8")
    full = _pytest(project, "--junitxml=junit.xml", fail=True)
    assert full.returncode == 1, full.stdout
    junit = project / "junit.xml"
    assert tor.parse_junit(junit.read_text(encoding="utf-8")) == {
        "tests.test_m::test_pass": tor.PASSED,
        "tests.test_m::test_always_fails": tor.FAILED,
        "tests.test_m::test_flaky": tor.FAILED,
        "tests.test_m::test_skip": tor.SKIPPED,
        "tests.test_m::test_xfail": tor.XFAILED,
        "tests.test_m::test_param[a.b]": tor.FAILED,
        "tests.test_m::test_param[c d]": tor.FAILED,
        "tests.test_m.TestGroup::test_in_class": tor.FAILED,
    }
    ids = tmp_path / "ids.txt"
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(project), "--out", str(ids)]) == 0
    assert ids.read_text(encoding="utf-8").splitlines() == [
        "tests/test_m.py::TestGroup::test_in_class",
        "tests/test_m.py::test_always_fails",
        "tests/test_m.py::test_flaky",
        "tests/test_m.py::test_param[a.b]",
        "tests/test_m.py::test_param[c d]",
    ]
    rerun = _pytest(project, f"@{ids}", "--junitxml=rerun.xml")
    assert rerun.returncode == 1, rerun.stdout
    out = tmp_path / "verdict.json"
    assert nc.main(["verdict", "--rc", "1", "--junit", str(junit), "--rerun", str(project / "rerun.xml"),
                    "--rerun-rc", "1", "--out", str(out)]) == 1
    verdict = json.loads(out.read_text(encoding="utf-8"))
    assert verdict["confirmed"] == ["tests.test_m.TestGroup::test_in_class", "tests.test_m::test_always_fails"]
    assert verdict["flaky"] == ["tests.test_m::test_flaky", "tests.test_m::test_param[a.b]",
                                "tests.test_m::test_param[c d]"]
    assert verdict["status"] == nc.RED


def test_a_real_collection_error_reaches_triage_with_the_module_to_collect(tmp_path: Path) -> None:
    """pytest stops with exit 2, and the report holds only the module's id."""
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']\n", encoding="utf-8")
    (project / "tests" / "test_m.py").write_text("def test_pass():\n    pass\n", encoding="utf-8")
    (project / "tests" / "test_broken.py").write_text("raise ImportError('regression')\n", encoding="utf-8")
    full = _pytest(project, "--junitxml=junit.xml")
    assert full.returncode == 2, full.stdout
    junit = project / "junit.xml"
    assert tor.parse_junit(junit.read_text(encoding="utf-8")) == {"::tests.test_broken": tor.ERROR}
    ids = tmp_path / "ids.txt"
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(project), "--out", str(ids)]) == 1
    assert not ids.exists()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "junit.xml").write_bytes(junit.read_bytes())
    assert nc.main(["verdict", "--rc", "2", "--junit", str(junit), "--out", str(bundle / "verdict.json")]) == 1
    clean = _bundle(tmp_path / "clean", {"tests.test_m::test_pass": tor.PASSED})
    record_file = tmp_path / "record.json"
    assert nc.main(["record", "--leg", f"3.13={bundle}", "--leg", f"3.12={clean}", "--sha", SHA2,
                    "--run-id", "9", "--run-attempt", "1", "--created-at", CREATED, "--out", str(record_file)]) == 0
    record = json.loads(record_file.read_text(encoding="utf-8"))
    assert record["nightly"]["status"] == {"3.12": nc.GREEN, "3.13": nc.RED}
    assert record["nightly"]["verdict"] == nc.RED
    green = tmp_path / "branch" / "runs" / "x"
    green.mkdir(parents=True)
    (green / "record.json").write_text(json.dumps(_green(SHA1, 1)), encoding="utf-8")
    (tmp_path / "branch" / "last-green.json").write_text(
        json.dumps({"canary": {"sha": SHA1, "path": "runs/x", "run_id": 1}, "legs": {}}), encoding="utf-8",
    )
    out = tmp_path / "triage.json"
    assert nc.main(["regressions", "--current", str(record_file), "--last-green-dir", str(tmp_path / "branch"),
                    "--repo-root", str(project), "--out", str(out)]) == 0
    triage = json.loads(out.read_text(encoding="utf-8"))
    assert triage["collection_errors"] == [
        {"test": "::tests.test_broken", "module": "tests/test_broken.py", "legs": ["3.13"]},
    ]
    assert triage["drift_check"]["legs"]["3.13"]["collection_errors"] == ["tests/test_broken.py"]
    assert triage["candidates"] == [] and triage["new_failing"] == []


# ---------------------------------------------------------------- verdict

@pytest.mark.parametrize(
    ("rc", "full", "rerun", "exit_status", "status", "flaky", "confirmed"),
    [
        (0, {_A: tor.PASSED}, None, 0, nc.GREEN, [], []),
        (1, {_A: tor.FAILED, _B: tor.PASSED}, {_A: tor.PASSED}, 0, nc.GREEN, [_A], []),
        (1, {_A: tor.FAILED}, {_A: tor.FAILED}, 1, nc.RED, [], [_A]),
        (1, {_A: tor.FAILED}, None, 1, nc.RED, [], [_A]),
        (1, {_A: tor.FAILED}, {}, 1, nc.RED, [], [_A]),
        (1, {_A: tor.FAILED}, {_A: tor.SKIPPED}, 1, nc.RED, [], [_A]),
        (1, {_A: tor.ERROR, _B: tor.XPASSED}, {_A: tor.PASSED, _B: tor.XPASSED}, 1, nc.RED, [_A], [_B]),
        (2, {"::tests.test_broken": tor.ERROR}, None, 1, nc.RED, [], ["::tests.test_broken"]),
        (2, {_A: tor.PASSED}, None, 1, nc.BROKEN, [], []),
        (5, {}, None, 1, nc.BROKEN, [], []),
        (0, {}, None, 1, nc.BROKEN, [], []),
        (1, {_A: tor.PASSED}, None, 1, nc.BROKEN, [], []),
        (1, {_A: tor.FAILED}, {_A: tor.PASSED}, 0, nc.GREEN, [_A], []),
    ],
    ids=["clean", "flaky", "confirmed", "no-rerun", "absent-from-rerun", "skipped-on-rerun",
         "flaky-and-confirmed", "collection-error", "interrupted", "no-tests", "rc0-no-tests", "rc1-nothing-red",
         "all-flaky"],
)
def test_verdict_sorts_flaky_from_confirmed_and_exits_on_anything_not_green(
    tmp_path: Path, rc: int, full: dict[str, str], rerun: dict[str, str] | None,
    exit_status: int, status: str, flaky: list[str], confirmed: list[str],
) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit(full), encoding="utf-8")
    argv = ["verdict", "--rc", str(rc), "--junit", str(junit), "--out", str(tmp_path / "verdict.json")]
    if rerun is not None:
        (tmp_path / "rerun.xml").write_text(_junit(rerun), encoding="utf-8")
        argv += ["--rerun", str(tmp_path / "rerun.xml"), "--rerun-rc", "1" if nc.red_ids(rerun) else "0"]
    assert nc.main(argv) == exit_status
    verdict = json.loads((tmp_path / "verdict.json").read_text(encoding="utf-8"))
    assert (verdict["status"], verdict["flaky"], verdict["confirmed"]) == (status, flaky, confirmed)


@pytest.mark.parametrize(("n", "mass"), [(50, False), (51, True)])
def test_a_mass_failure_ignores_the_rerun_and_confirms_every_red_test(n: int, mass: bool) -> None:
    full = {f"tests.test_a::test_{i:02}": tor.FAILED for i in range(n)}
    rerun = {t: tor.PASSED for t in full}
    verdict = nc.classify_leg(1, full, rerun)
    assert verdict["mass_failure"] is mass
    assert len(verdict["confirmed"]) == (n if mass else 0)
    assert len(verdict["flaky"]) == (0 if mass else n)


def _verdict_with_rerun(tmp_path: Path, rerun_rc: str, report: str | None) -> tuple[int, dict[str, Any]]:
    """`verdict` on one test red in the full run, with a rerun that exited
    `rerun_rc` and left `report` (None for no file)."""
    junit, rerun, out = tmp_path / "junit.xml", tmp_path / "rerun.xml", tmp_path / "verdict.json"
    junit.write_text(_junit({_A: tor.FAILED}), encoding="utf-8")
    if report is not None:
        rerun.write_text(report, encoding="utf-8")
    status = nc.main(["verdict", "--rc", "1", "--junit", str(junit), "--rerun", str(rerun),
                      "--rerun-rc", rerun_rc, "--out", str(out)])
    return status, json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("report", "reason"),
    [(None, "no readable report"), ("<testsuites><testcase", "no readable report"),
     ("<html></html>", "no readable report")],
    ids=["crashed", "truncated", "not-junit"],
)
def test_a_rerun_with_no_usable_report_keeps_the_red_tests_confirmed_and_writes_the_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], report: str | None, reason: str,
) -> None:
    status, verdict = _verdict_with_rerun(tmp_path, "137", report)
    assert status == 1
    assert (verdict["confirmed"], verdict["flaky"], verdict["status"]) == ([_A], [], nc.RED)
    assert reason in verdict["rerun_note"]
    assert reason in capsys.readouterr().out


@pytest.mark.parametrize("rerun_rc", [2, 3, 4, 5, 137])
def test_a_rerun_that_exited_other_than_0_or_1_keeps_the_red_tests_confirmed(tmp_path: Path, rerun_rc: int) -> None:
    """pytest exit 4 can leave a report in which a test passed; it does not
    make the test flaky."""
    status, verdict = _verdict_with_rerun(tmp_path, str(rerun_rc), _junit({_A: tor.PASSED}))
    assert status == 1
    assert (verdict["confirmed"], verdict["flaky"]) == ([_A], [])
    assert verdict["rerun_rc"] == rerun_rc
    assert f"exited {rerun_rc}" in verdict["rerun_note"]


@pytest.mark.parametrize(("rerun_rc", "flaky"), [("0", [_A]), ("1", [_A])])
def test_a_rerun_that_exited_0_or_1_counts(tmp_path: Path, rerun_rc: str, flaky: list[str]) -> None:
    status, verdict = _verdict_with_rerun(tmp_path, rerun_rc, _junit({_A: tor.PASSED}))
    assert status == 0
    assert verdict["flaky"] == flaky and verdict["rerun_note"] is None


@pytest.mark.parametrize("flags", [["--rerun", "r.xml"], ["--rerun-rc", "0"]], ids=["no-rc", "no-report"])
def test_the_rerun_flags_go_together(tmp_path: Path, flags: list[str]) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit({_A: tor.FAILED}), encoding="utf-8")
    assert nc.main(["verdict", "--rc", "1", "--junit", str(junit), *flags]) == 2


def test_verdict_dry_run_writes_nothing_and_keeps_the_exit_status(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit({_A: tor.FAILED}), encoding="utf-8")
    out = tmp_path / "verdict.json"
    assert nc.main(["verdict", "--rc", "1", "--junit", str(junit), "--out", str(out), "--dry-run"]) == 1
    assert not out.exists()


@pytest.mark.parametrize(
    "text",
    ["<html><body>502 Bad Gateway</body></html>", "<testcase classname='a' name='b'/>", "<x><testsuites/></x>"],
    ids=["html", "bare-testcase", "wrapped"],
)
def test_xml_that_is_not_a_junit_report_is_unreadable_for_both_readers(tmp_path: Path, text: str) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(text, encoding="utf-8")
    assert nc.main(["verdict", "--rc", "0", "--junit", str(junit)]) == 2
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(tmp_path),
                    "--out", str(tmp_path / "ids.txt")]) == 2


@pytest.mark.parametrize("root", ["testsuites", "testsuite"])
def test_a_report_with_either_junit_root_and_no_test_is_broken(tmp_path: Path, root: str) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(f"<{root}></{root}>", encoding="utf-8")
    out = tmp_path / "verdict.json"
    assert nc.main(["verdict", "--rc", "0", "--junit", str(junit), "--out", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["status"] == nc.BROKEN


def test_verdict_on_an_unreadable_report_is_a_usage_error(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    assert nc.main(["verdict", "--rc", "1", "--junit", str(junit)]) == 2


# ----------------------------------------------------------------- record

def _record_cmd(tmp_path: Path, *legs: str, extra: list[str] | None = None) -> tuple[int, dict[str, Any] | None]:
    out = tmp_path / "record.json"
    argv = ["record", *(a for leg in legs for a in ("--leg", leg)), "--sha", SHA1, "--run-id", "100",
            "--run-attempt", "2", "--event", "schedule", "--created-at", CREATED, "--out", str(out),
            *(extra or [])]
    rc = nc.main(argv)
    return rc, (json.loads(out.read_text(encoding="utf-8")) if out.exists() else None)


def test_the_record_is_format_1_and_test_outcome_record_reads_it(tmp_path: Path) -> None:
    b12 = _bundle(tmp_path / "b12", {_A: tor.PASSED, _B: tor.FAILED}, rerun={_B: tor.PASSED},
                  env={"python": "3.12.9", "sqlite_version": "3.45.1"})
    b13 = _bundle(tmp_path / "b13", {_A: tor.PASSED, _B: tor.PASSED})
    rc, record = _record_cmd(tmp_path, f"3.12={b12}", f"3.13={b13}:canary")
    assert rc == 0 and record is not None
    assert record["format"] == tor.RECORD_FORMAT == 1
    assert record["sha"] == SHA1
    assert set(record["legs"]) == {"3.12", "3.13"}
    assert record["legs"]["3.12"]["tests"] == {_A: tor.PASSED, _B: tor.FAILED}
    assert tor.is_baseline_record(record)
    diffs = tor.diff_records(record, record)
    assert [d.leg for d in diffs] == ["3.12", "3.13"]
    assert all(d.baseline_present and not d.violations for d in diffs)
    nightly = record["nightly"]
    assert nightly["run_id"] == 100 and nightly["run_attempt"] == 2 and nightly["event"] == "schedule"
    assert nightly["created_at"] == CREATED
    assert nightly["roles"] == {"3.12": "canary", "3.13": "canary"}
    assert nightly["env"]["3.12"] == {"python": "3.12.9", "sqlite_version": "3.45.1"}
    assert nightly["flaky"] == {"3.12": [_B], "3.13": []}
    assert nightly["confirmed"] == {"3.12": [], "3.13": []}
    assert nightly["verdict"] == nc.GREEN


def test_the_record_takes_each_legs_newest_attempt(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    _bundle(root / f"nightly-3.12-{SHA1}-a1", {_A: tor.FAILED})
    _bundle(root / f"nightly-3.12-{SHA1}-a2", {_A: tor.FAILED}, rerun={_A: tor.FAILED})
    _bundle(root / f"nightly-3.12-{SHA1}-a10", {_A: tor.PASSED})
    _bundle(root / f"nightly-3.12-{SHA2}-a11", {_A: tor.FAILED})
    _bundle(root / f"nightly-3.13-{SHA1}-a3", {_A: tor.FAILED})
    rc, record = _record_cmd(tmp_path, f"3.12={root}", f"3.1={root}", extra=["--canary-legs", "3.12"])
    assert rc == 0 and record is not None
    assert record["nightly"]["attempts"] == {"3.1": None, "3.12": 10}
    assert record["nightly"]["status"] == {"3.1": nc.MISSING, "3.12": nc.GREEN}
    assert record["legs"]["3.1"] is None


@pytest.mark.parametrize(
    ("legs", "verdict"),
    [
        ({"3.12": {_A: tor.PASSED}, "3.13": {_A: tor.PASSED}}, nc.GREEN),
        ({"3.12": {_A: tor.PASSED}, "3.13": {_A: tor.FAILED}}, nc.RED),
        ({"3.12": {_A: tor.PASSED}, "3.13": None}, nc.INCOMPLETE),
        ({"3.12": {_A: tor.PASSED}, "3.13": {_A: tor.PASSED}, "3.14": {_A: tor.FAILED}}, nc.GREEN),
        ({"3.12": {_A: tor.PASSED}, "3.13": {_A: tor.PASSED}, "3.14": None}, nc.GREEN),
    ],
    ids=["green", "red", "missing-canary", "red-extra", "missing-extra"],
)
def test_the_night_is_green_only_when_every_canary_leg_is(
    tmp_path: Path, legs: dict[str, dict[str, str] | None], verdict: str,
) -> None:
    specs = []
    for leg, outcomes in legs.items():
        directory = tmp_path / leg
        if outcomes is not None:
            _bundle(directory, outcomes)
        specs.append(f"{leg}={directory}" + (":extra" if leg == "3.14" else ""))
    rc, record = _record_cmd(tmp_path, *specs)
    assert rc == 0 and record is not None
    assert record["nightly"]["verdict"] == verdict


def test_the_record_says_why_a_rerun_did_not_count(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "b", {_A: tor.FAILED})
    verdict = nc.classify_leg(1, {_A: tor.FAILED}, {_A: tor.PASSED}, rerun_rc=4)
    (bundle / "verdict.json").write_text(json.dumps(verdict), encoding="utf-8")
    rc, record = _record_cmd(tmp_path, f"3.12={bundle}")
    assert rc == 0 and record is not None
    assert record["nightly"]["confirmed"]["3.12"] == [_A]
    assert record["nightly"]["notes"]["3.12"] == ["the rerun exited 4, so every red test stays confirmed"]


def test_one_named_canary_leg_is_not_a_green_night(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "b", {_A: tor.PASSED})
    rc, record = _record_cmd(tmp_path, f"3.12={bundle}")
    assert rc == 0 and record is not None
    nightly = record["nightly"]
    assert nightly["verdict"] == nc.INCOMPLETE
    assert nightly["status"] == {"3.12": nc.GREEN, "3.13": nc.MISSING}
    assert nightly["roles"]["3.13"] == nc.CANARY
    assert record["legs"]["3.13"] is None
    assert nightly["notes"]["3.13"] == ["no --leg names this canary leg"]


def test_the_flag_names_which_legs_a_green_night_needs(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "b", {_A: tor.PASSED})
    rc, record = _record_cmd(tmp_path, f"3.12={bundle}", extra=["--canary-legs", "3.12"])
    assert rc == 0 and record is not None
    assert record["nightly"]["verdict"] == nc.GREEN


def test_record_refuses_an_empty_flag(tmp_path: Path) -> None:
    """Otherwise one canary leg alone could read as a green night."""
    bundle = _bundle(tmp_path / "b", {_A: tor.PASSED})
    rc, record = _record_cmd(tmp_path, f"3.12={bundle}", extra=["--canary-legs", ""])
    assert rc == 2 and record is None


def test_a_required_leg_given_the_extra_role_is_not_a_green_night(tmp_path: Path) -> None:
    a, b = _bundle(tmp_path / "a", {_A: tor.PASSED}), _bundle(tmp_path / "b", {_A: tor.PASSED})
    rc, record = _record_cmd(tmp_path, f"3.12={a}", f"3.13={b}:extra")
    assert rc == 0 and record is not None
    assert record["nightly"]["verdict"] == nc.INCOMPLETE


def test_a_night_with_zero_tests_on_both_canary_legs_is_not_green(tmp_path: Path) -> None:
    a, b = _bundle(tmp_path / "a", {}), _bundle(tmp_path / "b", {})
    rc, record = _record_cmd(tmp_path, f"3.12={a}", f"3.13={b}")
    assert rc == 0 and record is not None
    assert record["nightly"]["status"] == {"3.12": nc.BROKEN, "3.13": nc.BROKEN}
    assert record["nightly"]["verdict"] == nc.RED


@pytest.mark.parametrize("required", [(), ("3.12", "3.13")], ids=["nothing-required", "default"])
def test_a_night_of_only_extra_legs_is_incomplete(required: tuple[str, ...]) -> None:
    """`all()` over no canary leg is True, which is the way this could read green."""
    assert nc.night_verdict({"3.14": nc.EXTRA}, {"3.14": nc.GREEN}, required) == nc.INCOMPLETE
    assert nc.night_verdict({}, {}, required) == nc.INCOMPLETE


def test_a_leg_where_every_test_skipped_is_broken_and_cannot_move_the_pointer(tmp_path: Path) -> None:
    """A missing dependency can skip the whole suite with exit 0."""
    assert nc.classify_leg(0, {_A: tor.SKIPPED, _B: tor.SKIPPED}, None)["status"] == nc.BROKEN
    assert nc.classify_leg(0, {_A: tor.SKIPPED, _B: tor.PASSED}, None)["status"] == nc.GREEN
    assert nc.classify_leg(0, {_A: tor.SKIPPED, _B: tor.XFAILED}, None)["status"] == nc.GREEN
    a = _bundle(tmp_path / "a", {_A: tor.SKIPPED}, rc=0)
    b = _bundle(tmp_path / "b", {_A: tor.PASSED})
    rc, record = _record_cmd(tmp_path, f"3.12={a}", f"3.13={b}")
    assert rc == 0 and record is not None
    assert record["nightly"]["status"]["3.12"] == nc.BROKEN
    assert record["nightly"]["verdict"] == nc.RED


def test_a_broken_canary_leg_makes_the_night_red(tmp_path: Path) -> None:
    _bundle(tmp_path / "3.12", {_A: tor.PASSED})
    _bundle(tmp_path / "3.13", {_A: tor.PASSED}, rc=2)
    rc, record = _record_cmd(tmp_path, f"3.12={tmp_path / '3.12'}", f"3.13={tmp_path / '3.13'}")
    assert rc == 0 and record is not None
    assert record["nightly"]["status"]["3.13"] == nc.BROKEN
    assert record["nightly"]["verdict"] == nc.RED


_BAD_FILES = [
    ("verdict.json", b"oops"),
    ("verdict.json", b"[]"),
    ("verdict.json", b"{}"),
    ("verdict.json", b'{"rc": 0, "red": "x", "flaky": [], "confirmed": [], "mass_failure": false, "status": "green"}'),
    ("verdict.json", b'{"rc": true, "red": [], "flaky": [], "confirmed": [], "mass_failure": false, "status": "green"}'),
    ("verdict.json", b'{"rc": 0, "red": [], "flaky": [], "confirmed": [], "status": "green"}'),
    ("verdict.json", b'{"rc": 0, "red": [], "flaky": null, "confirmed": [], "mass_failure": false, "status": "green"}'),
    ("verdict.json", b"\xff\xfe"),
    ("verdict.json", b"[" * 200_000),
    ("env.json", b"oops"),
    ("env.json", b"[]"),
    ("env.json", b"[" * 200_000),
    ("junit.xml", b"\xff\xfe"),
    ("junit.xml", b"<x"),
    ("junit.xml", b"<html></html>"),
]
_BAD_IDS = ["verdict-corrupt", "verdict-array", "verdict-empty", "verdict-red-not-a-list", "verdict-rc-bool",
            "verdict-no-mass-flag", "verdict-flaky-null", "verdict-not-utf8", "verdict-too-deep", "env-corrupt",
            "env-array", "env-too-deep", "junit-not-utf8",
            "junit-truncated", "junit-html"]


def _three_legs(tmp_path: Path) -> list[str]:
    for leg in ("3.12", "3.13", "3.14"):
        _bundle(tmp_path / leg, {_A: tor.PASSED})
    return [f"3.12={tmp_path / '3.12'}", f"3.13={tmp_path / '3.13'}", f"3.14={tmp_path / '3.14'}:extra"]


@pytest.mark.parametrize(("name", "data"), _BAD_FILES, ids=_BAD_IDS)
def test_a_bad_file_in_an_extra_leg_keeps_the_nights_green_record(tmp_path: Path, name: str, data: bytes) -> None:
    specs = _three_legs(tmp_path)
    (tmp_path / "3.14" / name).write_bytes(data)
    rc, record = _record_cmd(tmp_path, *specs)
    assert rc == 0 and record is not None
    assert record["nightly"]["status"] == {"3.12": nc.GREEN, "3.13": nc.GREEN, "3.14": nc.MISSING}
    assert record["nightly"]["verdict"] == nc.GREEN
    assert record["nightly"]["notes"]["3.14"]


@pytest.mark.parametrize(("name", "data"), _BAD_FILES, ids=_BAD_IDS)
def test_a_bad_file_in_a_canary_leg_reads_incomplete_with_the_record_written(
    tmp_path: Path, name: str, data: bytes,
) -> None:
    specs = _three_legs(tmp_path)
    (tmp_path / "3.13" / name).write_bytes(data)
    rc, record = _record_cmd(tmp_path, *specs)
    assert rc == 0 and record is not None
    assert record["nightly"]["status"]["3.13"] == nc.MISSING
    assert record["nightly"]["verdict"] == nc.INCOMPLETE
    assert record["nightly"]["notes"]["3.13"]


@pytest.mark.parametrize(
    ("method", "name"),
    [("is_file", "junit.xml"), ("exists", "env.json"), ("iterdir", None)],
    ids=["bundle-probe", "env-probe", "artifact-listing"],
)
def test_an_extra_leg_the_script_cannot_probe_keeps_the_nights_green_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, name: str | None,
) -> None:
    """A leg directory with mode 000 raises from a probe as well as a read, and
    which probe raises depends on the Python version."""
    specs = _three_legs(tmp_path)
    if method == "iterdir":
        (tmp_path / "3.14" / "env.json").unlink()
        (tmp_path / "3.14" / "junit.xml").unlink()
        (tmp_path / "3.14" / "verdict.json").unlink()
    real = getattr(Path, method)

    def denied(self: Path) -> Any:
        if (self.name == name and self.parent.name == "3.14") or (name is None and self.name == "3.14"):
            raise PermissionError(13, "denied", str(self))
        return real(self)

    monkeypatch.setattr(Path, method, denied)
    rc, record = _record_cmd(tmp_path, *specs)
    assert rc == 0 and record is not None
    assert record["nightly"]["status"] == {"3.12": nc.GREEN, "3.13": nc.GREEN, "3.14": nc.MISSING}
    assert record["nightly"]["verdict"] == nc.GREEN
    assert "denied" in " ".join(record["nightly"]["notes"]["3.14"])


def test_verdict_with_no_out_writes_nothing_and_warns_of_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit({_A: tor.PASSED}), encoding="utf-8")
    assert nc.main(["verdict", "--rc", "0", "--junit", str(junit)]) == 0
    out = capsys.readouterr().out
    assert "::warning::" not in out and "leg green" in out
    assert sorted(p.name for p in tmp_path.iterdir()) == ["junit.xml"]


def test_a_non_ascii_test_id_survives_an_ascii_stdout(tmp_path: Path) -> None:
    """Run as a process, since the encoding belongs to the interpreter's stream."""
    junit = tmp_path / "junit.xml"
    test = "tests.test_a::test_caf\u00e9"
    junit.write_text(_junit({test: tor.FAILED}), encoding="utf-8")
    out = tmp_path / "v.json"
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "verdict", "--rc", "1", "--junit", str(junit), "--out", str(out)],
        env={**os.environ, "PYTHONIOENCODING": "ascii", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, timeout=60,
    )
    assert r.returncode == 1, r.stderr
    assert b"confirmed: tests.test_a::test_caf\\xe9" in r.stdout
    assert json.loads(out.read_text(encoding="utf-8"))["confirmed"] == [test]


def test_the_verdict_file_is_written_before_anything_is_printed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step that dies on its output must not lose what `record` needs."""
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit({_A: tor.FAILED}), encoding="utf-8")
    out = tmp_path / "v.json"

    real_print = print

    def broken_pipe(*args: Any, **kwargs: Any) -> None:
        if "file" not in kwargs:
            raise BrokenPipeError(32, "pipe")
        real_print(*args, **kwargs)

    monkeypatch.setattr("builtins.print", broken_pipe)
    assert nc.main(["verdict", "--rc", "1", "--junit", str(junit), "--out", str(out)]) == 2
    assert json.loads(out.read_text(encoding="utf-8"))["confirmed"] == [_A]


def test_a_rerun_report_too_deep_to_parse_leaves_every_red_test_confirmed(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit({_A: tor.FAILED}), encoding="utf-8")
    rerun = tmp_path / "rerun.xml"
    rerun.write_text("<testsuites>" + "<x>" * 200_000 + "</x>" * 200_000 + "</testsuites>", encoding="utf-8")
    out = tmp_path / "v.json"
    assert nc.main(["verdict", "--rc", "1", "--junit", str(junit), "--rerun", str(rerun), "--rerun-rc", "1",
                    "--out", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["confirmed"] == [_A]


@pytest.mark.parametrize(
    "report",
    [_junit({_A: tor.FAILED}), _junit({_B: tor.PASSED, _A: tor.ERROR}), _junit({})],
    ids=["failing-test", "erroring-test", "no-tests"],
)
def test_a_green_verdict_over_a_report_that_is_not_green_reads_incomplete(tmp_path: Path, report: str) -> None:
    specs = _three_legs(tmp_path)
    (tmp_path / "3.13" / "junit.xml").write_text(report, encoding="utf-8")
    rc, record = _record_cmd(tmp_path, *specs)
    assert rc == 0 and record is not None
    assert record["nightly"]["status"]["3.13"] == nc.MISSING
    assert record["nightly"]["verdict"] == nc.INCOMPLETE
    assert "verdict.json" in record["nightly"]["notes"]["3.13"][0]


def test_a_verdict_that_marks_a_red_test_flaky_without_listing_it_red_is_refused() -> None:
    outcomes = {_A: tor.FAILED}
    rerun = {_A: tor.PASSED}
    verdict = nc.classify_leg(1, outcomes, rerun)
    assert nc.check_verdict(verdict, outcomes, rerun) is None
    assert nc.check_verdict({**verdict, "flaky": [], "confirmed": []}, outcomes, rerun) is not None
    assert nc.check_verdict({**verdict, "status": nc.RED}, outcomes, rerun) is not None
    assert nc.check_verdict({**verdict, "red": []}, outcomes, rerun) is not None


def _flaky_verdict(**fields: Any) -> dict[str, Any]:
    """A verdict over `_A` failing, marked flaky, with these fields replaced."""
    return {**nc.classify_leg(1, {_A: tor.FAILED}, {_A: tor.PASSED}), **fields}


@pytest.mark.parametrize(
    ("fields", "rerun", "says"),
    [
        ({}, None, "no readable rerun.xml"),
        ({}, {_B: tor.PASSED}, "does not show passing"),
        ({}, {_A: tor.FAILED}, "does not show passing"),
        ({"mass_failure": True}, {_A: tor.PASSED}, "mass failure"),
        ({"rc": 0}, {_A: tor.PASSED}, "exited 0"),
        ({"rerun_rc": 4}, {_A: tor.PASSED}, "a rerun that exited 4"),
    ],
    ids=["no-rerun-report", "rerun-lacks-it", "rerun-failed-it", "mass-failure", "rc-0-over-red", "rerun-exit-4"],
)
def test_a_flaky_list_needs_a_rerun_that_shows_those_tests_passing(
    fields: dict[str, Any], rerun: dict[str, str] | None, says: str,
) -> None:
    problem = nc.check_verdict(_flaky_verdict(**fields), {_A: tor.FAILED}, rerun)
    assert problem is not None and says in problem


def test_rc_0_over_a_red_report_is_refused_with_nothing_flaky(tmp_path: Path) -> None:
    verdict = {**nc.classify_leg(1, {_A: tor.FAILED}, None), "rc": 0}
    assert "exited 0" in (nc.check_verdict(verdict, {_A: tor.FAILED}, None) or "")


def test_a_forged_flaky_list_reads_as_a_missing_leg_not_a_green_one(tmp_path: Path) -> None:
    forged = _bundle(tmp_path / "3.13", {_A: tor.FAILED})
    (forged / "verdict.json").write_text(json.dumps(_flaky_verdict()), encoding="utf-8")
    good = _bundle(tmp_path / "3.12", {_A: tor.PASSED})
    rc, record = _record_cmd(tmp_path, f"3.12={good}", f"3.13={forged}")
    assert rc == 0 and record is not None
    assert record["nightly"]["status"]["3.13"] == nc.MISSING
    assert record["nightly"]["verdict"] == nc.INCOMPLETE
    assert "no readable rerun.xml" in record["nightly"]["notes"]["3.13"][0]


def test_a_bundle_whose_rerun_shows_the_flaky_test_passing_reads_green(tmp_path: Path) -> None:
    flaky = _bundle(tmp_path / "3.13", {_A: tor.FAILED}, rerun={_A: tor.PASSED})
    good = _bundle(tmp_path / "3.12", {_A: tor.PASSED})
    rc, record = _record_cmd(tmp_path, f"3.12={good}", f"3.13={flaky}")
    assert rc == 0 and record is not None
    assert record["nightly"]["flaky"]["3.13"] == [_A]
    assert record["nightly"]["verdict"] == nc.GREEN


def test_a_verdict_rc_that_is_a_boolean_is_refused() -> None:
    """False equals 0, so the status check alone would accept it."""
    outcomes = {_A: tor.PASSED}
    assert nc.check_verdict({**nc.classify_leg(0, outcomes, None), "rc": False}, outcomes, None) == "verdict.json holds no pytest rc"


def test_an_unreadable_report_beside_an_empty_verdict_is_a_missing_leg(tmp_path: Path) -> None:
    """The verdict is dropped with the report, so nothing reads its fields."""
    specs = _three_legs(tmp_path)
    (tmp_path / "3.13" / "junit.xml").write_bytes(b"\xff\xfe")
    (tmp_path / "3.13" / "verdict.json").write_bytes(b"{}")
    rc, record = _record_cmd(tmp_path, *specs)
    assert rc == 0 and record is not None
    assert record["nightly"]["status"]["3.13"] == nc.MISSING


def test_a_verdict_with_no_report_is_a_missing_leg() -> None:
    verdict = nc.classify_leg(3, {_A: tor.PASSED}, None)
    record = nc.build_nightly_record(
        {"3.12": None}, {"3.12": verdict}, {"3.12": None}, {"3.12": nc.CANARY}, {"3.12": None},
        SHA1, 1, 1, "schedule", CREATED, required=("3.12",),
    )
    assert record["nightly"]["status"]["3.12"] == nc.MISSING
    assert record["nightly"]["verdict"] == nc.INCOMPLETE
    assert record["nightly"]["rc"]["3.12"] == 3
    assert record["nightly"]["mass_failure"]["3.12"] is False


def test_the_record_holds_each_legs_rc_and_mass_flag() -> None:
    many = {f"t::{i}": tor.FAILED for i in range(60)}
    record = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": (many, None), "3.14": (None, None)},
                    roles={"3.14": nc.EXTRA})
    assert record["nightly"]["rc"] == {"3.12": 0, "3.13": 1, "3.14": None}
    assert record["nightly"]["mass_failure"] == {"3.12": False, "3.13": True, "3.14": False}


def test_an_absent_env_file_is_a_null_environment_not_a_bad_leg(tmp_path: Path) -> None:
    specs = _three_legs(tmp_path)
    (tmp_path / "3.13" / "env.json").unlink()
    rc, record = _record_cmd(tmp_path, *specs)
    assert rc == 0 and record is not None
    assert record["nightly"]["status"]["3.13"] == nc.GREEN
    assert record["nightly"]["env"]["3.13"] is None


def test_a_leg_whose_report_does_not_parse_or_has_no_verdict_is_missing(tmp_path: Path) -> None:
    bad = _bundle(tmp_path / "3.12", {_A: tor.PASSED})
    (bad / "junit.xml").write_text("<testsuites><testcase", encoding="utf-8")
    no_verdict = _bundle(tmp_path / "3.13", {_A: tor.PASSED})
    (no_verdict / "verdict.json").unlink()
    rc, record = _record_cmd(tmp_path, f"3.12={bad}", f"3.13={no_verdict}")
    assert rc == 0 and record is not None
    assert record["nightly"]["status"] == {"3.12": nc.MISSING, "3.13": nc.MISSING}
    assert record["nightly"]["verdict"] == nc.INCOMPLETE


@pytest.mark.parametrize(
    ("spec", "parsed"),
    [
        ("3.12=dir", ("3.12", "dir", "canary")),
        ("3.14=a/b:extra", ("3.14", "a/b", "extra")),
        ("3.13=a/b:canary", ("3.13", "a/b", "canary")),
        ("3.12=C:\\runner\\bundle", ("3.12", "C:\\runner\\bundle", "canary")),
        ("3.12=a:b", ("3.12", "a:b", "canary")),
    ],
)
def test_a_leg_spec_names_a_role_only_by_a_role_suffix(spec: str, parsed: tuple[str, str, str]) -> None:
    assert nc.parse_leg_spec(spec) == parsed


@pytest.mark.parametrize("legs", [("3.12",), ("=dir",), ("3.12=",), ("3.12=a", "3.12=b")])
def test_a_bad_or_repeated_leg_is_a_usage_error(tmp_path: Path, legs: tuple[str, ...]) -> None:
    rc, record = _record_cmd(tmp_path, *legs)
    assert rc == 2 and record is None


@pytest.mark.parametrize("created_at", ["Nov 4", "2026/11/04", "20261104", "2026-W45-3", ""],
                         ids=["word", "slashes", "iso-basic", "iso-week", "empty-is-now"])
def test_a_created_at_with_no_date_is_a_usage_error(tmp_path: Path, created_at: str) -> None:
    bundle = _bundle(tmp_path / "b", {_A: tor.PASSED})
    out = tmp_path / "r.json"
    status = nc.main(["record", "--leg", f"3.12={bundle}", "--sha", SHA1, "--run-id", "1", "--run-attempt", "1",
                      "--created-at", created_at, "--out", str(out)])
    if created_at:
        assert status == 2 and not out.exists()
    else:
        assert status == 0 and out.exists()


@pytest.mark.parametrize(
    "bad",
    [["--sha", "notasha"], ["--sha", "abc123"], ["--sha", "--evil"], ["--sha", "A" * 40],
     ["--run-id", "0"], ["--run-id", "-7"], ["--run-attempt", "0"], ["--run-attempt", "-1"],
     ["--created-at", "9999-99-99"], ["--created-at", "2026-02-30T00:00:00Z"]],
    ids=["sha-word", "sha-too-short", "sha-flag", "sha-upper", "run-0", "run-negative", "attempt-0", "attempt-negative",
         "date-impossible", "date-no-such-day"],
)
def test_record_refuses_a_run_it_could_not_publish(tmp_path: Path, bad: list[str]) -> None:
    bundle = _bundle(tmp_path / "b", {_A: tor.PASSED})
    out = tmp_path / "r.json"
    args = {"--sha": SHA1, "--run-id": "1", "--run-attempt": "1", "--created-at": CREATED, bad[0]: bad[1]}
    assert nc.main(["record", "--leg", f"3.12={bundle}", *[f"{k}={v}" for k, v in args.items()], "--out", str(out)]) == 2
    assert not out.exists()


def test_record_takes_the_shortest_commit_id_and_the_first_run(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "b", {_A: tor.PASSED})
    out = tmp_path / "r.json"
    assert nc.main(["record", "--leg", f"3.12={bundle}", "--sha", "abc1234", "--run-id", "1", "--run-attempt", "1",
                    "--created-at", CREATED, "--canary-legs", "3.12", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["sha"] == "abc1234"


def test_record_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _bundle(tmp_path / "b", {_A: tor.FAILED})
    rc, record = _record_cmd(tmp_path, f"3.12={tmp_path / 'b'}", extra=["--dry-run", "--canary-legs", "3.12"])
    assert rc == 0 and record is None
    assert "night: red" in capsys.readouterr().out


# ------------------------------------------------------------ regressions

_LEGS = ["3.12", "3.13"]


def test_a_flaky_failure_on_the_last_green_night_counts_as_passed() -> None:
    green = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.FAILED}, {_A: tor.PASSED})})
    assert green["nightly"]["verdict"] == nc.GREEN
    tonight = _night(SHA2, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.FAILED}, {_A: tor.FAILED})})
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["candidates"] == [{"test": _A, "node_id": None, "legs": ["3.13"], "before": {"3.13": tor.PASSED}}]
    assert triage["new_failing"] == []
    assert triage["events"] == []


def test_a_failure_on_the_last_green_night_that_was_not_flaky_keeps_its_outcome() -> None:
    """`baseline_outcome` rewrites a failure only when the record says it was flaky."""
    green = _night(SHA1, {"3.13": ({_A: tor.FAILED}, {_A: tor.PASSED})})
    green["nightly"]["flaky"]["3.13"] = []
    assert nc.baseline_outcome(green, "3.13", _A) == tor.FAILED


def test_a_new_test_that_fails_is_new_failing_and_skips_the_drift_check() -> None:
    green = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.PASSED}, None)})
    tonight = _night(SHA2, {
        "3.12": ({_A: tor.FAILED, _C: tor.FAILED}, None),
        "3.13": ({_A: tor.PASSED, _C: tor.FAILED}, None),
    })
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["new_failing"] == [{"test": _C, "node_id": None, "legs": ["3.12", "3.13"]}]
    assert [c["test"] for c in triage["candidates"]] == [_A]
    assert triage["drift_check"] == {
        "sha": SHA1,
        "legs": {"3.12": {"tests": [_A], "node_ids": [], "unmapped": [_A], "collection_errors": []}},
    }


def test_a_test_skipped_on_the_last_green_night_that_now_fails_is_a_candidate() -> None:
    green = _night(SHA1, {"3.12": ({_A: tor.SKIPPED}, None), "3.13": ({_A: tor.PASSED}, None)})
    tonight = _night(SHA2, {"3.12": ({_A: tor.FAILED}, None), "3.13": ({_A: tor.PASSED}, None)})
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["candidates"] == [{"test": _A, "node_id": None, "legs": ["3.12"], "before": {"3.12": tor.SKIPPED}}]


@pytest.mark.parametrize("listed", [_LEGS, [*_LEGS, "3.14", "macos"]])
def test_extra_legs_never_produce_an_event_or_a_candidate(listed: list[str]) -> None:
    roles = {"3.14": nc.EXTRA, "macos": nc.EXTRA}
    green = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.PASSED}, None),
                          "3.14": ({_A: tor.PASSED}, None), "macos": ({_A: tor.PASSED}, None)}, roles)
    tonight = _night(SHA2, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.PASSED}, None),
                            "3.14": ({_A: tor.FAILED, _C: tor.FAILED}, None), "macos": (None, None)}, roles)
    assert tonight["nightly"]["verdict"] == nc.GREEN
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, listed)
    assert triage["canary_legs"] == _LEGS
    assert (triage["events"], triage["candidates"], triage["new_failing"], triage["drift_check"]) == ([], [], [], None)
    no_baseline = nc.find_regressions(tonight, None, None, listed)
    assert no_baseline["events"] == []


def test_a_leg_the_flag_leaves_out_is_not_a_canary_leg() -> None:
    tonight = _night(SHA2, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.FAILED}, None)})
    triage = nc.find_regressions(tonight, None, None, ["3.12"])
    assert triage["events"] == []


def test_no_last_green_night_is_one_no_baseline_event_with_every_confirmed_test() -> None:
    tonight = _night(SHA2, {"3.12": ({_A: tor.FAILED, _B: tor.FAILED}, {_B: tor.PASSED}),
                            "3.13": ({_C: tor.FAILED}, None)})
    triage = nc.find_regressions(tonight, None, None, _LEGS)
    [event] = triage["events"]
    assert event["failures"] == [{"test": _A, "node_id": None, "legs": ["3.12"]},
                                 {"test": _C, "node_id": None, "legs": ["3.13"]}]
    assert {k: v for k, v in event.items() if k != "failures"} == {
        "key": f"no-baseline-{SHA2[:12]}", "kind": "no-baseline", "legs": ["3.12", "3.13"], "tests": [_A, _C],
        "reason": "no last green night yet",
    }
    assert triage["candidates"] == [] and triage["drift_check"] is None


@pytest.mark.parametrize("kind", ["no-baseline", "drift"])
def test_a_night_with_no_comparison_still_says_which_leg_failed_what(tmp_path: Path, kind: str) -> None:
    root = _tree(tmp_path, "tests/test_a.py", "tests/test_b.py")
    before = {_A: tor.PASSED, _C: tor.PASSED}
    green = _night(SHA2, {"3.12": (before, None), "3.13": (before, None)})
    tonight = _night(SHA2, {"3.12": ({_A: tor.FAILED, _C: tor.FAILED}, None), "3.13": ({_A: tor.PASSED, _C: tor.FAILED}, None)})
    triage = nc.find_regressions(tonight, green if kind == "drift" else None,
                                 {"sha": SHA2, "path": "runs/x"} if kind == "drift" else None,
                                 _LEGS, lambda rel: (root / rel).is_file())
    assert triage["events"][0]["kind"] == kind
    assert triage["events"][0]["failures"] == [
        {"test": _A, "node_id": "tests/test_a.py::test_one", "legs": ["3.12"]},
        {"test": _C, "node_id": "tests/test_b.py::test_three", "legs": ["3.12", "3.13"]},
    ]


def test_a_red_night_on_the_last_green_commit_is_drift() -> None:
    green = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.PASSED}, None)})
    tonight = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.FAILED}, None)}, run_id=101)
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["events"] == [{"key": f"drift-{SHA1[:12]}", "kind": "drift", "legs": ["3.13"], "tests": [_A],
                                 "failures": [{"test": _A, "node_id": None, "legs": ["3.13"]}]}]
    assert triage["candidates"] == [] and triage["drift_check"] is None


def test_a_green_night_has_nothing_to_triage() -> None:
    green = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.FAILED}, {_A: tor.PASSED})})
    triage = nc.find_regressions(green, None, None, _LEGS)
    assert (triage["events"], triage["candidates"], triage["new_failing"]) == ([], [], [])


@pytest.mark.parametrize(
    ("leg_13", "rc"),
    [((None, None), 0), (({}, None), 5)],
    ids=["missing", "no-tests"],
)
def test_a_missing_or_broken_canary_leg_is_canary_broken(
    leg_13: tuple[dict[str, str] | None, None], rc: int,
) -> None:
    tonight = _night(SHA2, {"3.12": ({_A: tor.PASSED}, None), "3.13": leg_13}, rcs={"3.13": rc})
    assert tonight["nightly"]["status"]["3.13"] in (nc.MISSING, nc.BROKEN)
    triage = nc.find_regressions(tonight, None, None, _LEGS)
    assert triage["events"] == [{"key": "canary-broken", "kind": "canary-broken", "legs": ["3.13"], "tests": [],
                                 "failures": []}]


def test_a_canary_leg_with_a_bad_verdict_names_the_failures_its_report_holds(tmp_path: Path) -> None:
    """Its verdict is gone, so nothing is confirmed, but `junit.xml` still reads."""
    root = _tree(tmp_path, "tests/test_a.py")
    bad = _bundle(tmp_path / "3.13", {_A: tor.FAILED, _B: tor.PASSED})
    (bad / "verdict.json").write_text("{bad", encoding="utf-8")
    good = _bundle(tmp_path / "3.12", {_A: tor.PASSED, _B: tor.PASSED})
    rc, record = _record_cmd(tmp_path, f"3.12={good}", f"3.13={bad}")
    assert rc == 0 and record is not None
    assert record["nightly"]["status"]["3.13"] == nc.MISSING
    triage = nc.find_regressions(record, None, None, _LEGS, lambda rel: (root / rel).is_file())
    assert triage["events"] == [{
        "key": "canary-broken", "kind": "canary-broken", "legs": ["3.13"], "tests": [_A],
        "failures": [{"test": _A, "node_id": "tests/test_a.py::test_one", "legs": ["3.13"]}],
    }]


_BROKEN_MODULE = "::tests.test_broken"


def _collection_error_night() -> tuple[dict[str, Any], dict[str, Any]]:
    """A last green night, and a night whose 3.13 leg stopped on an import error."""
    green = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.PASSED}, None)})
    tonight = _night(SHA2, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_BROKEN_MODULE: tor.ERROR}, None)},
                     rcs={"3.13": 2})
    return green, tonight


def test_a_collection_error_is_its_own_entry_with_the_module_to_collect(tmp_path: Path) -> None:
    """pytest stopped with exit 2: the suite did not run, nothing passed or failed."""
    root = _tree(tmp_path, "tests/test_broken.py")
    green, tonight = _collection_error_night()
    assert tonight["nightly"]["verdict"] == nc.RED
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS,
                                 lambda rel: (root / rel).is_file())
    assert triage["collection_errors"] == [
        {"test": _BROKEN_MODULE, "module": "tests/test_broken.py", "legs": ["3.13"]},
    ]
    assert (triage["candidates"], triage["new_failing"]) == ([], [])
    assert triage["events"] == [{"key": f"collection-error-{SHA2[:12]}", "kind": "collection-error",
                                 "legs": ["3.13"], "tests": [_BROKEN_MODULE]}]
    assert triage["drift_check"]["legs"]["3.13"]["collection_errors"] == ["tests/test_broken.py"]


def test_a_collection_error_with_no_file_keeps_its_entry_and_its_event() -> None:
    green, tonight = _collection_error_night()
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["collection_errors"] == [{"test": _BROKEN_MODULE, "module": None, "legs": ["3.13"]}]
    assert triage["drift_check"] is None
    assert triage["events"][0]["tests"] == [_BROKEN_MODULE]


def test_a_real_broken_nested_conftest_gets_its_directory_and_an_event(tmp_path: Path) -> None:
    """pytest names the directory, `::tests.sub`, which no `.py` file matches."""
    project = tmp_path / "project"
    (project / "tests" / "sub").mkdir(parents=True)
    (project / "pyproject.toml").write_text("[tool.pytest.ini_options]\ntestpaths = ['tests']\n", encoding="utf-8")
    (project / "tests" / "test_m.py").write_text("def test_pass():\n    pass\n", encoding="utf-8")
    (project / "tests" / "sub" / "conftest.py").write_text("raise ImportError('regression')\n", encoding="utf-8")
    (project / "tests" / "sub" / "test_s.py").write_text("def test_s():\n    pass\n", encoding="utf-8")
    full = _pytest(project, "--junitxml=junit.xml")
    assert full.returncode == 2, full.stdout
    outcomes = tor.parse_junit((project / "junit.xml").read_text(encoding="utf-8"))
    assert outcomes == {"::tests.sub": tor.ERROR}
    green = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.PASSED}, None)})
    tonight = _night(SHA2, {"3.12": ({_A: tor.PASSED}, None), "3.13": (outcomes, None)}, rcs={"3.13": 2})
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS,
                                 lambda rel: (project / rel).is_file(), None, lambda rel: (project / rel).is_dir())
    assert triage["collection_errors"] == [{"test": "::tests.sub", "module": "tests/sub/", "legs": ["3.13"]}]
    assert triage["events"][0]["kind"] == "collection-error"
    assert triage["drift_check"]["legs"]["3.13"]["collection_errors"] == ["tests/sub/"]
    # Named on the command line, the conftest fails at startup: exit 4, not 2.
    assert _pytest(project, "--collect-only", "-q", "tests/sub/").returncode == 4
    (project / "tests" / "sub" / "conftest.py").write_text("", encoding="utf-8")
    assert _pytest(project, "--collect-only", "-q", "tests/sub/").returncode == 0


def test_regressions_finds_the_directory_of_a_collection_error(tmp_path: Path) -> None:
    root = _tree(tmp_path / "repo", "tests/sub/test_s.py")
    tonight = _night(SHA2, {"3.12": ({"::tests.sub": tor.ERROR}, None), "3.13": ({_A: tor.PASSED}, None)}, rcs={"3.12": 2})
    current = tmp_path / "record.json"
    current.write_text(json.dumps(tonight), encoding="utf-8")
    (tmp_path / "branch").mkdir()
    out = tmp_path / "t.json"
    assert nc.main(["regressions", "--current", str(current), "--last-green-dir", str(tmp_path / "branch"),
                    "--repo-root", str(root), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["collection_errors"][0]["module"] == "tests/sub/"


def test_a_collection_error_with_no_baseline_is_listed_and_in_the_event(tmp_path: Path) -> None:
    root = _tree(tmp_path, "tests/test_broken.py")
    _, tonight = _collection_error_night()
    triage = nc.find_regressions(tonight, None, None, _LEGS, lambda rel: (root / rel).is_file())
    assert triage["events"][0]["kind"] == "no-baseline" and triage["events"][0]["tests"] == [_BROKEN_MODULE]
    assert triage["collection_errors"][0]["module"] == "tests/test_broken.py"


def test_a_collection_error_beside_ordinary_failures_splits_them(tmp_path: Path) -> None:
    """With `--continue-on-collection-errors` both appear in one report."""
    root = _tree(tmp_path, "tests/test_broken.py", "tests/test_a.py")
    green = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.PASSED}, None)})
    tonight = _night(SHA2, {"3.12": ({_A: tor.PASSED}, None),
                            "3.13": ({_A: tor.FAILED, _BROKEN_MODULE: tor.ERROR}, None)})
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS,
                                 lambda rel: (root / rel).is_file())
    assert [c["test"] for c in triage["candidates"]] == [_A]
    assert [e["test"] for e in triage["collection_errors"]] == [_BROKEN_MODULE]
    assert triage["new_failing"] == []
    leg = triage["drift_check"]["legs"]["3.13"]
    assert leg["node_ids"] == ["tests/test_a.py::test_one"]
    assert leg["collection_errors"] == ["tests/test_broken.py"]


def test_a_mass_failure_drift_checks_an_even_sample_of_twenty() -> None:
    tests = {f"tests.test_a::test_{i:02}": tor.PASSED for i in range(60)}
    green = _night(SHA1, {"3.12": (tests, None), "3.13": (tests, None)})
    failing = {t: tor.FAILED for t in tests}
    tonight = _night(SHA2, {"3.12": (failing, None), "3.13": (failing, None)})
    assert tonight["nightly"]["mass_failure"] == {"3.12": True, "3.13": True}
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["mass_failure"] is True
    assert len(triage["candidates"]) == 60
    sample = [f"tests.test_a::test_{i:02}" for i in range(0, 60, 3)]
    legs = triage["drift_check"]["legs"]
    assert legs["3.12"]["tests"] == sample and legs["3.13"]["tests"] == sample


def test_a_mass_failure_on_one_leg_samples_that_leg_and_marks_the_night() -> None:
    """A sample taken over both legs held none of the tests that failed on the
    other leg's Python, so the drift check there proved nothing."""
    tests = {f"tests.test_a::test_{i:02}": tor.PASSED for i in range(60)}
    only_313 = "tests.test_z::test_only_313"
    everything = {**tests, only_313: tor.PASSED}
    green = _night(SHA1, {"3.12": (everything, None), "3.13": (everything, None)})
    tonight = _night(SHA2, {
        "3.12": ({**everything, **{t: tor.FAILED for t in tests}}, None),
        "3.13": ({**everything, only_313: tor.FAILED}, None),
    })
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert tonight["nightly"]["mass_failure"] == {"3.12": True, "3.13": False}
    assert triage["mass_failure"] is True
    legs = triage["drift_check"]["legs"]
    assert legs["3.13"]["tests"] == [only_313]
    assert len(legs["3.12"]["tests"]) == 20 and set(legs["3.12"]["tests"]) <= set(tests)


def test_a_mass_failure_on_one_leg_does_not_sample_the_other() -> None:
    tests = {f"tests.test_a::test_{i:02}": tor.PASSED for i in range(60)}
    green = _night(SHA1, {"3.12": (tests, None), "3.13": (tests, None)})
    thirty = [f"tests.test_a::test_{i:02}" for i in range(30)]
    tonight = _night(SHA2, {
        "3.12": ({t: tor.FAILED for t in tests}, None),
        "3.13": ({**tests, **{t: tor.FAILED for t in thirty}}, None),
    })
    assert tonight["nightly"]["mass_failure"] == {"3.12": True, "3.13": False}
    legs = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)["drift_check"]["legs"]
    assert len(legs["3.12"]["tests"]) == 20
    assert legs["3.13"]["tests"] == thirty


def test_a_short_list_is_the_whole_sample_in_order() -> None:
    assert nc.even_sample(["b", "a"], 20) == ["a", "b"]
    items = [f"t{i:02}" for i in range(20)]
    assert nc.even_sample(items, 20) == items


def test_a_mass_failures_drift_check_names_only_the_sample(tmp_path: Path) -> None:
    root = _tree(tmp_path, "tests/test_a.py")
    tests = {f"tests.test_a::test_{i:02}": tor.PASSED for i in range(60)}
    gone = "tests.test_gone::test_x"
    green = _night(SHA1, {"3.12": ({**tests, gone: tor.PASSED}, None), "3.13": (tests, None)})
    tonight = _night(SHA2, {"3.12": ({**{t: tor.FAILED for t in tests}, gone: tor.FAILED}, None), "3.13": (tests, None)})
    leg = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS,
                              lambda rel: (root / rel).is_file())["drift_check"]["legs"]["3.12"]
    assert len(leg["tests"]) == 20
    assert len(leg["node_ids"]) + len(leg["unmapped"]) == 20


def test_a_leg_collects_only_the_collection_errors_it_had(tmp_path: Path) -> None:
    root = _tree(tmp_path, "tests/test_one.py", "tests/test_two.py")
    green = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.PASSED}, None)})
    tonight = _night(SHA2, {"3.12": ({"::tests.test_one": tor.ERROR}, None), "3.13": ({"::tests.test_two": tor.ERROR}, None)},
                     rcs={"3.12": 2, "3.13": 2})
    legs = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS,
                               lambda rel: (root / rel).is_file())["drift_check"]["legs"]
    assert legs["3.12"]["collection_errors"] == ["tests/test_one.py"]
    assert legs["3.13"]["collection_errors"] == ["tests/test_two.py"]


def test_a_listed_leg_the_record_does_not_hold_is_a_broken_canary_leg_with_no_mass_failure() -> None:
    """`record --canary-legs 3.13` then `regressions` with the default: 3.12 has no
    role, status or mass flag there, and must read as missing, not as green."""
    tonight = _night(SHA2, {"3.13": ({_A: tor.PASSED}, None)})
    for key in ("roles", "status", "mass_failure", "confirmed", "flaky", "rc", "env", "attempts", "notes"):
        tonight["nightly"][key].pop("3.12", None)
    triage = nc.find_regressions(tonight, None, None, _LEGS)
    assert triage["canary_legs"] == _LEGS
    assert triage["events"][0]["legs"] == ["3.12"]
    assert triage["mass_failure"] is False


def test_the_sample_does_not_depend_on_input_order() -> None:
    items = [f"t::{i:03}" for i in range(137)]
    assert nc.even_sample(list(reversed(items)), 20) == nc.even_sample(items, 20)
    assert len(set(nc.even_sample(items, 20))) == 20


def test_each_test_is_drift_checked_on_the_leg_that_failed_it() -> None:
    green = _night(SHA1, {"3.12": ({_A: tor.PASSED, _B: tor.PASSED, _C: tor.PASSED}, None),
                          "3.13": ({_A: tor.PASSED, _B: tor.PASSED, _C: tor.PASSED}, None)})
    tonight = _night(SHA2, {"3.12": ({_A: tor.FAILED, _B: tor.PASSED, _C: tor.FAILED}, None),
                            "3.13": ({_A: tor.PASSED, _B: tor.FAILED, _C: tor.FAILED}, None)})
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["mass_failure"] is False
    legs = triage["drift_check"]["legs"]
    assert (legs["3.12"]["tests"], legs["3.13"]["tests"]) == ([_A, _C], [_B, _C])


def test_the_drift_check_hands_pytest_node_ids_and_names_what_has_none(tmp_path: Path) -> None:
    """pytest refuses a JUnit id with exit 4, so `tests` alone could not be run."""
    root = _tree(tmp_path, "tests/test_a.py", "tests/test_b.py")
    gone = "tests.test_gone::test_x"
    before = {_A: tor.PASSED, _C: tor.PASSED, gone: tor.PASSED}
    green = _night(SHA1, {"3.12": (before, None), "3.13": (before, None)})
    tonight = _night(SHA2, {"3.12": ({**before, _A: tor.FAILED, gone: tor.FAILED, "tests.test_b::test_new": tor.FAILED}, None),
                            "3.13": (before, None)})
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS,
                                 lambda rel: (root / rel).is_file())
    assert triage["drift_check"]["legs"]["3.12"] == {
        "tests": [_A, gone], "node_ids": ["tests/test_a.py::test_one"], "unmapped": [gone], "collection_errors": [],
    }
    assert {c["test"]: c["node_id"] for c in triage["candidates"]} == {_A: "tests/test_a.py::test_one", gone: None}
    assert triage["new_failing"][0]["node_id"] == "tests/test_b.py::test_new"


def test_below_the_threshold_every_candidate_is_drift_checked() -> None:
    tests = {f"tests.test_a::test_{i:02}": tor.PASSED for i in range(30)}
    green = _night(SHA1, {"3.12": (tests, None), "3.13": (tests, None)})
    tonight = _night(SHA2, {"3.12": (tests, None), "3.13": ({t: tor.FAILED for t in tests}, None)})
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["mass_failure"] is False
    assert list(triage["drift_check"]["legs"]) == ["3.13"]
    assert triage["drift_check"]["legs"]["3.13"]["tests"] == sorted(tests)


def test_regressions_dry_run_writes_nothing(tmp_path: Path) -> None:
    current = tmp_path / "record.json"
    current.write_text(json.dumps(_night(SHA2, {"3.12": ({_A: tor.FAILED}, None)})), encoding="utf-8")
    out = tmp_path / "triage.json"
    (tmp_path / "branch").mkdir()
    assert nc.main(["regressions", "--current", str(current), "--last-green-dir", str(tmp_path / "branch"),
                    "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()


@pytest.mark.parametrize("flag", ["3.12,3.13", "3.13", ""], ids=["third-leg-unlisted", "second-leg-unlisted", "empty"])
def test_regressions_refuses_a_flag_that_would_hide_a_canary_leg(tmp_path: Path, flag: str) -> None:
    """The night reads red over that leg and `events` would be empty with exit 0."""
    legs = {"3.12": ({_A: tor.FAILED}, None), "3.13": ({_A: tor.PASSED}, None)}
    if flag == "3.12,3.13":
        legs["3.14"] = ({_A: tor.FAILED}, None)
    current = tmp_path / "record.json"
    current.write_text(json.dumps(_night(SHA2, legs)), encoding="utf-8")
    (tmp_path / "branch").mkdir()
    out = tmp_path / "t.json"
    assert nc.main(["regressions", "--current", str(current), "--last-green-dir", str(tmp_path / "branch"),
                    "--canary-legs", flag, "--out", str(out)]) == 2
    assert not out.exists()


def test_a_listed_leg_the_record_marks_extra_is_not_refused(tmp_path: Path) -> None:
    legs = {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.PASSED}, None), "3.14": ({_A: tor.FAILED}, None)}
    current = tmp_path / "record.json"
    current.write_text(json.dumps(_night(SHA2, legs, roles={"3.14": nc.EXTRA})), encoding="utf-8")
    (tmp_path / "branch").mkdir()
    out = tmp_path / "t.json"
    assert nc.main(["regressions", "--current", str(current), "--last-green-dir", str(tmp_path / "branch"),
                    "--canary-legs", "3.12,3.13", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["events"] == []


def test_a_pointer_file_with_no_canary_entry_yet_is_the_night_before_the_first_green_one(tmp_path: Path) -> None:
    """A first red night writes `{"canary": null, "legs": {}}`."""
    triage = _regress_on(tmp_path, json.dumps({"canary": None, "legs": {}}), None)
    assert triage["events"][0]["reason"] == "no last green night yet"
    assert triage["last_green"] is None


def test_a_last_green_dir_that_does_not_exist_is_a_usage_error(tmp_path: Path) -> None:
    """A mistyped path must not read as the night before the first green one."""
    current = tmp_path / "record.json"
    current.write_text(json.dumps(_red(SHA2, 100)), encoding="utf-8")
    out = tmp_path / "triage.json"
    assert nc.main(["regressions", "--current", str(current), "--last-green-dir", str(tmp_path / "typo"),
                    "--out", str(out)]) == 2
    assert not out.exists()


def _regress_on(tmp_path: Path, pointer: str, record: bytes | None) -> dict[str, Any]:
    """`regressions` for a red night over a results branch with this
    `last-green.json` and, when given, this record at `runs/x`."""
    current = tmp_path / "record.json"
    current.write_text(json.dumps(_red(SHA2, 100)), encoding="utf-8")
    branch = tmp_path / "branch"
    (branch / "runs" / "x").mkdir(parents=True)
    (branch / "last-green.json").write_text(pointer, encoding="utf-8")
    if record is not None:
        (branch / "runs" / "x" / "record.json").write_bytes(record)
    out = tmp_path / "t.json"
    assert nc.main(["regressions", "--current", str(current), "--last-green-dir", str(branch), "--out", str(out)]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


_POINTS_AT_X = json.dumps({"canary": {"sha": SHA1, "path": "runs/x", "run_id": 1, "attempt": 1}, "legs": {}})


@pytest.mark.parametrize(
    ("record", "why"),
    [(None, "holds no record"), (b"oops", "unreadable"), (b"\xff\xfe", "unreadable"), (b"[" * 200_000, "unreadable")],
    ids=["gone", "corrupt", "not-utf8", "too-deep"],
)
def test_a_last_green_record_that_cannot_be_read_is_a_no_baseline_event_with_the_reason(
    tmp_path: Path, record: bytes | None, why: str,
) -> None:
    """Only a green night moves the pointer, so a lasting regression that lost
    its triage here would never produce one."""
    triage = _regress_on(tmp_path, _POINTS_AT_X, record)
    [event] = triage["events"]
    assert (event["key"], event["kind"]) == (f"no-baseline-{SHA2[:12]}", "no-baseline")
    assert event["tests"] == [_A]
    assert "runs/x" in event["reason"] and why in event["reason"]
    assert triage["last_green"] is None


# ----------------------------------------------------------------- publish

def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "HOME": str(repo.parent), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    r = subprocess.run(
        ["git", "-c", "user.name=canary", "-c", "user.email=canary@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=repo, env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    return r.stdout.strip()


def _make_main_repo(repo: Path) -> tuple[Path, list[str], str]:
    """A main branch c0 -> c1 -> c2, and a commit off c0 that is not on it."""
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    shas = []
    for i in range(3):
        _git(repo, "commit", "-q", "--allow-empty", "-m", f"c{i}")
        shas.append(_git(repo, "rev-parse", "HEAD"))
    side = _git(repo, "commit-tree", f"{shas[0]}^{{tree}}", "-p", shas[0], "-m", "side")
    return repo, shas, side


@pytest.fixture(scope="module")
def main_repo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[str], str]:
    """One repository for the module; the tests read it. A test that moves a ref
    builds its own with `_make_main_repo`."""
    return _make_main_repo(tmp_path_factory.mktemp("main-repo") / "main")


def _publish(tmp_path: Path, repo: Path, record: dict[str, Any], triage: dict[str, Any] | None,
             *extra: str) -> int:
    """`publish` on a checkout of the results branch at `tmp_path/branch`."""
    branch = tmp_path / "branch"
    branch.mkdir(exist_ok=True)
    rec, tri = tmp_path / "record.json", tmp_path / "triage.json"
    rec.write_text(json.dumps(record), encoding="utf-8")
    if triage is None:
        tri.unlink(missing_ok=True)
    else:
        tri.write_text(json.dumps(triage), encoding="utf-8")
    return nc.main(["publish", "--record", str(rec), "--triage", str(tri), "--branch-dir", str(branch),
                    "--main-repo", str(repo), *extra])


def _green(sha: str, run_id: int, day: int = 4, run_attempt: int = 1) -> dict[str, Any]:
    return _night(sha, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.PASSED}, None)},
                  run_id=run_id, run_attempt=run_attempt, created_at=f"2026-11-{day:02}T02:17:00Z")


def _red(sha: str, run_id: int, day: int = 4, run_attempt: int = 1) -> dict[str, Any]:
    return _night(sha, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.FAILED}, None)},
                  run_id=run_id, run_attempt=run_attempt, created_at=f"2026-11-{day:02}T02:17:00Z")


def _index(tmp_path: Path) -> list[dict[str, Any]]:
    text = (tmp_path / "branch" / "index.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _last_green(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "branch" / "last-green.json").read_text(encoding="utf-8"))


def test_publish_writes_the_run_the_index_and_the_last_green_pointer(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, shas, _ = main_repo
    record = _green(shas[0], 100)
    triage = nc.find_regressions(record, None, None, _LEGS)
    triage["events"] = [{"key": "pr-1786", "kind": "pr", "legs": [], "tests": []}]
    assert _publish(tmp_path, repo, record, triage) == 0
    branch = tmp_path / "branch"
    assert sorted(p.relative_to(branch).as_posix() for p in branch.rglob("*") if p.is_file()) == [
        "index.jsonl", "last-green.json", "runs/2026-11-04-100-a1/record.json", "runs/2026-11-04-100-a1/triage.json",
    ]
    assert json.loads((branch / "runs/2026-11-04-100-a1/record.json").read_text(encoding="utf-8")) == record
    [line] = _index(tmp_path)
    assert line == {
        "run_id": 100, "run_attempt": 1, "date": "2026-11-04", "path": "runs/2026-11-04-100-a1", "sha": shas[0],
        "verdict": nc.GREEN, "status": {"3.12": nc.GREEN, "3.13": nc.GREEN},
        "counts": {leg: record["legs"][leg]["counts"] for leg in _LEGS},
        "flaky": {"3.12": [], "3.13": []}, "events": ["pr-1786"],
    }
    pointer = {"sha": shas[0], "path": "runs/2026-11-04-100-a1", "run_id": 100, "attempt": 1, "date": "2026-11-04"}
    assert _last_green(tmp_path) == {"canary": pointer, "legs": {"3.12": pointer, "3.13": pointer}}


def test_a_rerun_of_the_same_run_leaves_one_index_line(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, shas, _ = main_repo
    assert _publish(tmp_path, repo, _red(shas[1], 100, run_attempt=1), None) == 0
    assert _publish(tmp_path, repo, _red(shas[1], 100, run_attempt=2), None) == 0
    assert [(e["run_id"], e["run_attempt"]) for e in _index(tmp_path)] == [(100, 2)]
    assert _publish(tmp_path, repo, _red(shas[2], 101, day=5), None) == 0
    assert _publish(tmp_path, repo, _red(shas[1], 100, run_attempt=3), None) == 0
    assert [(e["run_id"], e["run_attempt"]) for e in _index(tmp_path)] == [(100, 3), (101, 1)]
    assert _index(tmp_path)[0]["path"] == "runs/2026-11-04-100-a3"
    # An older attempt published late leaves the newest attempt's line.
    assert _publish(tmp_path, repo, _red(shas[1], 100, run_attempt=2), None) == 0
    assert [(e["run_id"], e["run_attempt"]) for e in _index(tmp_path)] == [(100, 3), (101, 1)]


def test_a_red_reattempt_leaves_last_green_on_the_green_record_unchanged(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    """Attempt 1 is green and moves the pointer; attempt 2 of the same run
    is red. The pointer still names attempt 1's record, which still exists,
    is byte-for-byte what attempt 1 wrote, and is green."""
    repo, (c0, _, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, run_attempt=1), None) == 0
    pointer = _last_green(tmp_path)["canary"]
    green_file = tmp_path / "branch" / pointer["path"] / "record.json"
    green_bytes = green_file.read_bytes()
    assert _publish(tmp_path, repo, _red(c0, 100, run_attempt=2), None) == 0
    assert _last_green(tmp_path)["canary"] == pointer
    assert green_file.read_bytes() == green_bytes
    assert json.loads(green_bytes)["nightly"]["verdict"] == nc.GREEN
    assert (tmp_path / "branch/runs/2026-11-04-100-a2/record.json").is_file()
    [line] = _index(tmp_path)
    assert (line["run_attempt"], line["verdict"], line["path"]) == (2, nc.RED, "runs/2026-11-04-100-a2")
    # regressions reads the green attempt as the last green night.
    current = tmp_path / "tonight.json"
    current.write_text(json.dumps(_red(c0, 100, run_attempt=2)), encoding="utf-8")
    out = tmp_path / "triage-out.json"
    assert nc.main(["regressions", "--current", str(current), "--last-green-dir", str(tmp_path / "branch"),
                    "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["events"][0]["key"] == f"drift-{c0[:12]}"


def test_the_same_attempt_published_again_replaces_its_own_index_line() -> None:
    lines = nc.merge_index([], nc.index_entry(_red(SHA1, 100), None))
    again = nc.merge_index(lines, nc.index_entry(_green(SHA1, 100), None))
    assert [json.loads(line)["verdict"] for line in again] == [nc.GREEN]


def test_the_last_green_pointer_moves_only_on_green_and_only_forward(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, c1, c2), side = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, day=1), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c0
    assert _publish(tmp_path, repo, _red(c1, 101, day=2), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c0
    # The red night's 3.12 leg was green, and a red night changes no entry.
    assert _last_green(tmp_path)["legs"]["3.12"]["sha"] == c0
    assert _last_green(tmp_path)["legs"]["3.13"]["sha"] == c0
    assert _publish(tmp_path, repo, _green(c2, 102, day=3), None) == 0
    assert _last_green(tmp_path)["canary"] == {"sha": c2, "path": "runs/2026-11-03-102-a1", "run_id": 102,
                                               "attempt": 1, "date": "2026-11-03"}
    assert _publish(tmp_path, repo, _green(c1, 103, day=4), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c2
    assert _publish(tmp_path, repo, _green(side, 104, day=5), None) == 1
    assert _last_green(tmp_path)["canary"]["sha"] == c2
    assert _publish(tmp_path, repo, _green(c2, 105, day=6), None) == 0
    assert _last_green(tmp_path)["canary"]["run_id"] == 105


def test_a_late_older_attempt_at_the_same_commit_does_not_take_the_pointer(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, run_attempt=2), None) == 0
    assert _publish(tmp_path, repo, _green(c0, 100, run_attempt=1), None) == 0
    assert _last_green(tmp_path)["canary"]["path"] == "runs/2026-11-04-100-a2"
    assert _last_green(tmp_path)["legs"]["3.13"]["path"] == "runs/2026-11-04-100-a2"
    assert _index(tmp_path)[0]["run_attempt"] == 2


@pytest.mark.parametrize("night", [_green, _red], ids=["green", "red"])
def test_the_same_run_and_attempt_on_a_later_date_writes_nothing(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str], night: Any,
) -> None:
    """`created_at` defaults to now, so a republish lands in a new directory,
    which a refusal keyed on the path would not see."""
    repo, (c0, _, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, day=4), None) == 0
    branch = tmp_path / "branch"
    before = {p: p.read_bytes() for p in branch.rglob("*") if p.is_file()}
    assert _publish(tmp_path, repo, night(c0, 100, day=5), None) == 1
    assert {p: p.read_bytes() for p in branch.rglob("*") if p.is_file()} == before
    assert "::error::last green names runs/2026-11-04-100-a1" in capsys.readouterr().out


def test_a_refusal_needs_the_same_attempt_as_well_as_the_same_run(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, run_attempt=1), None) == 0
    assert _publish(tmp_path, repo, _red(c0, 100, run_attempt=2), None) == 0
    assert (tmp_path / "branch/runs/2026-11-04-100-a2/record.json").is_file()


def test_named_runs_reads_the_canary_entry_and_the_leg_entries() -> None:
    a = {"sha": SHA1, "path": "runs/a", "run_id": 5, "attempt": 2}
    b = {"sha": SHA1, "path": "runs/b", "run_id": 6, "attempt": 1}
    assert nc.named_runs({"canary": a, "legs": {"3.12": b}}, (5, 2)) == ["runs/a"]
    assert nc.named_runs({"canary": a, "legs": {"3.12": b}}, (6, 1)) == ["runs/b"]
    assert nc.named_runs({"canary": None, "legs": {}}, (5, 2)) == []


def test_a_late_green_attempt_cannot_take_an_empty_pointer_after_a_newer_red_one(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    assert _publish(tmp_path, repo, _red(c0, 100, run_attempt=2), None) == 0
    assert _publish(tmp_path, repo, _green(c0, 100, run_attempt=1), None) == 0
    assert _last_green(tmp_path) == {"canary": None, "legs": {}}
    assert "nothing moved" in capsys.readouterr().out
    assert [line["run_attempt"] for line in _index(tmp_path)] == [2]


def test_a_late_green_run_cannot_take_a_pointer_after_a_newer_red_run_of_its_commit(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, c1, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 90, day=1), None) == 0
    assert _publish(tmp_path, repo, _red(c1, 101, day=3), None) == 0
    assert _publish(tmp_path, repo, _green(c1, 100, day=2), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c0


def test_a_late_run_of_another_commit_is_not_superseded(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    """Run 101 is newer but tested a different commit, so it says nothing about c0."""
    repo, (c0, c1, _), _ = main_repo
    assert _publish(tmp_path, repo, _red(c1, 101, day=3), None) == 0
    assert _publish(tmp_path, repo, _green(c0, 100, day=2), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c0


def test_superseded_at_commit_reads_each_part_of_the_order() -> None:
    lines = nc.merge_index([], nc.index_entry(_red(SHA1, 100, run_attempt=2), None))
    assert nc.superseded_at_commit(lines, _green(SHA1, 100, run_attempt=1)) is True
    assert nc.superseded_at_commit(lines, _green(SHA1, 99, run_attempt=9)) is True
    assert nc.superseded_at_commit(lines, _green(SHA1, 100, run_attempt=2)) is False
    assert nc.superseded_at_commit(lines, _green(SHA1, 100, run_attempt=3)) is False
    assert nc.superseded_at_commit(lines, _green(SHA2, 99)) is False
    assert nc.superseded_at_commit(["", *lines], _green(SHA2, 99)) is False


def test_a_late_older_run_at_the_same_commit_does_not_take_the_pointer(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 101, day=5), None) == 0
    assert _publish(tmp_path, repo, _green(c0, 100, day=4), None) == 0
    assert _last_green(tmp_path)["canary"]["run_id"] == 101


def test_a_newer_attempt_at_the_same_commit_takes_the_pointer(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, run_attempt=1), None) == 0
    assert _publish(tmp_path, repo, _green(c0, 100, run_attempt=2), None) == 0
    assert _last_green(tmp_path)["canary"]["path"] == "runs/2026-11-04-100-a2"


def test_a_red_republish_of_the_run_and_attempt_last_green_names_writes_nothing(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100), None) == 0
    branch = tmp_path / "branch"
    before = {p: p.read_bytes() for p in branch.rglob("*") if p.is_file()}
    assert _publish(tmp_path, repo, _red(c0, 100), None) == 1
    assert {p: p.read_bytes() for p in branch.rglob("*") if p.is_file()} == before
    assert "::error::last green names runs/2026-11-04-100-a1" in capsys.readouterr().out


def _with_extra(sha: str, run_id: int, extra: str, day: int) -> dict[str, Any]:
    """A night with green canary legs and an extra 3.14 leg with this outcome."""
    return _night(sha, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.PASSED}, None), "3.14": ({_A: extra}, None)},
                  roles={"3.14": nc.EXTRA}, run_id=run_id, created_at=f"2026-11-{day:02}T02:17:00Z")


def test_a_leg_entry_protects_its_record_as_well(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    """The extra leg went red, so its entry alone still names the first night."""
    repo, (c0, c1, _), _ = main_repo
    assert _publish(tmp_path, repo, _with_extra(c0, 100, tor.PASSED, 1), None) == 0
    assert _publish(tmp_path, repo, _with_extra(c1, 101, tor.FAILED, 2), None) == 0
    green = _last_green(tmp_path)
    assert (green["canary"]["run_id"], green["legs"]["3.14"]["run_id"]) == (101, 100)
    assert _publish(tmp_path, repo, _red(c0, 100, day=1), None) == 1


def test_republishing_the_same_content_is_a_no_op(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    record = _green(c0, 100)
    assert _publish(tmp_path, repo, record, None) == 0
    pointer = _last_green(tmp_path)
    assert _publish(tmp_path, repo, record, None) == 0
    assert _last_green(tmp_path) == pointer
    assert len(_index(tmp_path)) == 1


def _tree_bytes(tmp_path: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in (tmp_path / "branch").rglob("*") if p.is_file()}


_PR_1 = {"events": [{"key": "pr-1"}]}


def test_republishing_is_byte_for_byte_a_no_op_with_or_without_triage(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, c1, _), _ = main_repo
    first = _red(c0, 100, day=1)
    assert _publish(tmp_path, repo, first, _PR_1) == 0
    assert _publish(tmp_path, repo, _green(c1, 101, day=2), None) == 0
    before = _tree_bytes(tmp_path)
    # An older run: the index order must not change.
    assert _publish(tmp_path, repo, first, _PR_1) == 0
    assert _tree_bytes(tmp_path) == before
    # No triage this time: the events on its index line and its triage file stay.
    assert _publish(tmp_path, repo, first, None) == 0
    assert _tree_bytes(tmp_path) == before
    assert _index(tmp_path)[0]["events"] == ["pr-1"]


def test_republishing_a_pointed_record_with_another_triage_keeps_the_first(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    record = _green(c0, 100)
    assert _publish(tmp_path, repo, record, _PR_1) == 0
    before = _tree_bytes(tmp_path)
    assert _publish(tmp_path, repo, record, {"events": [{"key": "pr-2"}]}) == 0
    assert _tree_bytes(tmp_path) == before


def test_a_triage_can_still_join_a_record_published_without_one(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    record = _red(c0, 100)
    assert _publish(tmp_path, repo, record, None) == 0
    assert _publish(tmp_path, repo, record, _PR_1) == 0
    assert (tmp_path / "branch/runs/2026-11-04-100-a1/triage.json").is_file()
    assert _index(tmp_path)[0]["events"] == ["pr-1"]


def test_a_changed_unpointed_record_takes_its_new_triage_and_events(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    both_red = _night(c0, {"3.12": ({_A: tor.FAILED}, None), "3.13": ({_A: tor.FAILED}, None)}, run_id=100)
    assert _publish(tmp_path, repo, both_red, _PR_1) == 0
    assert _publish(tmp_path, repo, _red(c0, 100), {"events": [{"key": "pr-2"}]}) == 0
    assert _index(tmp_path)[0]["events"] == ["pr-2"]
    assert json.loads((tmp_path / "branch/runs/2026-11-04-100-a1/triage.json").read_text(encoding="utf-8")) == {
        "events": [{"key": "pr-2"}],
    }


def test_a_changed_unpointed_record_with_no_triage_keeps_the_runs_events(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    both_red = _night(c0, {"3.12": ({_A: tor.FAILED}, None), "3.13": ({_A: tor.FAILED}, None)}, run_id=100)
    assert _publish(tmp_path, repo, both_red, _PR_1) == 0
    assert _publish(tmp_path, repo, _red(c0, 100), None) == 0
    assert _index(tmp_path)[0]["events"] == ["pr-1"]
    assert (tmp_path / "branch/runs/2026-11-04-100-a1/triage.json").is_file()


def test_a_new_attempt_does_not_inherit_the_events_of_the_last_one() -> None:
    lines = nc.merge_index([], {**nc.index_entry(_red(SHA1, 100, run_attempt=1), None), "events": ["pr-1"]})
    merged = nc.merge_index(lines, nc.index_entry(_red(SHA1, 100, run_attempt=2), None), keep_events=True)
    assert json.loads(merged[0])["events"] == []
    late = nc.merge_index(merged, nc.index_entry(_red(SHA1, 100, run_attempt=1), None), keep_events=True)
    assert late == merged


def test_the_index_is_ordered_by_run_and_skips_blank_lines() -> None:
    lines = nc.merge_index([], nc.index_entry(_red(SHA1, 9), None))
    lines = nc.merge_index(["", *lines, "  "], nc.index_entry(_red(SHA1, 10), None))
    lines = nc.merge_index(lines, nc.index_entry(_red(SHA1, 2), None))
    assert [json.loads(line)["run_id"] for line in lines] == [2, 9, 10]


def test_republishing_a_run_no_entry_names_replaces_it(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    """Every leg red, so no entry points here and a workflow retry may rewrite it."""
    repo, (c0, _, _), _ = main_repo
    both_red = _night(c0, {"3.12": ({_A: tor.FAILED}, None), "3.13": ({_A: tor.FAILED}, None)}, run_id=100)
    assert _publish(tmp_path, repo, both_red, None) == 0
    assert _publish(tmp_path, repo, _green(c0, 100), None) == 0
    published = json.loads((tmp_path / "branch/runs/2026-11-04-100-a1/record.json").read_text(encoding="utf-8"))
    assert published["nightly"]["verdict"] == nc.GREEN
    assert _index(tmp_path)[0]["verdict"] == nc.GREEN


def test_a_pointer_whose_record_is_gone_exits_1_until_a_green_night_moves_it(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, (c0, c1, c2), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, day=1), None) == 0
    (tmp_path / "branch/runs/2026-11-01-100-a1/record.json").unlink()
    both_red = _night(c1, {"3.12": ({_A: tor.FAILED}, None), "3.13": ({_A: tor.FAILED}, None)},
                      run_id=101, created_at="2026-11-02T02:17:00Z")
    assert _publish(tmp_path, repo, both_red, None) == 1
    assert "names runs/2026-11-01-100-a1, which holds no record" in capsys.readouterr().out
    assert _publish(tmp_path, repo, _green(c2, 102, day=3), None) == 0
    assert "::error::" not in capsys.readouterr().out


def _point_at(tmp_path: Path, run_dir: str, sha: str, legs: tuple[str, ...] = ()) -> None:
    """Hand-edit `last-green.json` so the canary entry, or these leg entries
    alone, name a run directory."""
    entry = {"sha": sha, "path": run_dir, "run_id": 1, "attempt": 1}
    green = {"canary": None if legs else entry, "legs": {leg: entry for leg in legs}}
    (tmp_path / "branch" / "last-green.json").write_text(json.dumps(green), encoding="utf-8")


def test_a_pointer_naming_a_red_record_exits_1_until_a_green_night_moves_it(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, (c0, c1, c2), _ = main_repo
    assert _publish(tmp_path, repo, _red(c0, 1, day=1), None) == 0
    _point_at(tmp_path, "runs/2026-11-01-1-a1", c0)
    capsys.readouterr()
    assert _publish(tmp_path, repo, _red(c1, 2, day=2), None) == 1
    assert "canary: names runs/2026-11-01-1-a1, which reads red there" in capsys.readouterr().out
    assert _publish(tmp_path, repo, _green(c2, 3, day=3), None) == 0
    assert "::error::" not in capsys.readouterr().out


def test_a_leg_entry_naming_a_record_where_that_leg_was_red_exits_1(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str],
) -> None:
    """`_red` leaves 3.12 green and 3.13 red, so only the 3.13 entry is wrong."""
    repo, (c0, c1, _), _ = main_repo
    assert _publish(tmp_path, repo, _red(c0, 1, day=1), None) == 0
    _point_at(tmp_path, "runs/2026-11-01-1-a1", c0, legs=("3.12", "3.13"))
    capsys.readouterr()
    assert _publish(tmp_path, repo, _red(c1, 2, day=2), None) == 1
    out = capsys.readouterr().out
    assert "leg 3.13: names runs/2026-11-01-1-a1, which reads red there" in out
    assert "leg 3.12: names" not in out


@pytest.mark.parametrize("content", [b"not json", b"[]", b"\xff\xfe", b"[" * 200_000],
                         ids=["corrupt", "array", "not-utf8", "too-deep"])
def test_a_pointer_naming_a_corrupt_record_exits_1(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str], content: bytes,
) -> None:
    repo, (c0, c1, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 1, day=1), None) == 0
    (tmp_path / "branch/runs/2026-11-01-1-a1/record.json").write_bytes(content)
    capsys.readouterr()
    assert _publish(tmp_path, repo, _red(c1, 2, day=2), None) == 1
    assert "canary: names runs/2026-11-01-1-a1, whose record is unreadable" in capsys.readouterr().out
    assert _publish(tmp_path, repo, _red(c1, 2, day=2), None, "--dry-run") == 1


def test_a_pointer_naming_a_red_record_is_no_baseline(tmp_path: Path) -> None:
    triage = _regress_on(tmp_path, _POINTS_AT_X, json.dumps(_red(SHA1, 1)).encode())
    assert triage["events"][0]["kind"] == "no-baseline"
    assert triage["events"][0]["reason"] == "last-green.json names runs/x, a night that reads red, not green"


def test_a_good_pointer_reports_no_record_problem() -> None:
    green_record = _green(SHA1, 1)
    entry = {"sha": SHA1, "path": "runs/a", "run_id": 1}
    roles = green_record["nightly"]["roles"]
    assert nc.record_problems({"canary": entry, "legs": {"3.12": entry}}, lambda p: (green_record, None), roles) == []
    assert nc.record_problems({}, lambda p: (None, "never read"), roles) == []


def test_a_retired_legs_entry_exits_0_with_one_notice_each_night(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str],
) -> None:
    """No night names the leg, so it could never move and exit 1 would never clear."""
    repo, (c0, c1, _), side = main_repo
    branch = tmp_path / "branch"
    branch.mkdir()
    retired = {"sha": side, "path": "runs/gone", "run_id": 1, "attempt": 1}
    (branch / "last-green.json").write_text(json.dumps({"canary": None, "legs": {"3.11": retired}}), encoding="utf-8")
    for run_id, sha in ((2, c0), (3, c1)):
        capsys.readouterr()
        assert _publish(tmp_path, repo, _green(sha, run_id, day=run_id), None) == 0
        out = capsys.readouterr().out
        assert out.count("::notice::last green leg 3.11") == 1 and "::error::" not in out
    assert _last_green(tmp_path)["legs"]["3.11"] == retired


def test_a_leg_tonight_names_is_checked_even_when_it_is_extra(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, (c0, _, _), side = main_repo
    branch = tmp_path / "branch"
    branch.mkdir()
    entry = {"sha": side, "path": "runs/gone", "run_id": 1, "attempt": 1}
    (branch / "last-green.json").write_text(json.dumps({"canary": None, "legs": {"3.14": entry}}), encoding="utf-8")
    assert _publish(tmp_path, repo, _with_extra(c0, 2, tor.FAILED, 2), None) == 1
    out = capsys.readouterr().out
    assert "leg 3.14: names runs/gone" in out and f"leg 3.14: names {side[:12]}" in out


@pytest.mark.parametrize(("answer", "says"), [(False, "is not on main"), (None, "could not be checked against main")],
                         ids=["not-on-main", "cannot-place"])
def test_a_pointer_main_does_not_place_is_a_problem_saying_which_way(answer: bool | None, says: str) -> None:
    entry = {"sha": SHA1, "path": "runs/a", "run_id": 1}
    problems = nc.off_main_pointers({"canary": entry, "legs": {"3.12": entry}}, lambda sha: answer, {"3.12": nc.CANARY})
    assert len(problems) == 2
    assert problems[0].startswith(f"canary: names {SHA1[:12]}, which {says};")
    assert problems[1].startswith(f"leg 3.12: names {SHA1[:12]}, which {says};")
    assert nc.off_main_pointers({"canary": entry, "legs": {}}, lambda sha: True, {}) == []


def test_an_entry_with_no_attempt_counts_as_older_than_the_first_attempt() -> None:
    old = {"canary": {"sha": SHA1, "path": "runs/a", "run_id": 5}, "legs": {}}
    new, _, _ = nc.next_last_green(old, _green(SHA1, 5, run_attempt=1), lambda a, b: True, lambda sha: True)
    assert new["canary"]["attempt"] == 1


def test_a_branch_dir_that_does_not_exist_is_a_usage_error(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    rec = tmp_path / "record.json"
    rec.write_text(json.dumps(_green(c0, 100)), encoding="utf-8")
    missing = tmp_path / "typo"
    assert nc.main(["publish", "--record", str(rec), "--triage", str(tmp_path / "t.json"), "--branch-dir", str(missing),
                    "--main-repo", str(repo)]) == 2
    assert not missing.exists()


def test_the_record_this_publish_writes_counts_as_present(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    """The first pointer names a directory that exists only after the writes."""
    repo, (c0, _, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100), None, "--dry-run") == 0
    assert _publish(tmp_path, repo, _green(c0, 100), None) == 0


def test_a_night_that_moves_nothing_leaves_the_pointer_file_untouched(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, c1, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, day=1), None) == 0
    green_file = tmp_path / "branch" / "last-green.json"
    spaced = json.dumps(json.loads(green_file.read_text(encoding="utf-8")), indent=4)
    green_file.write_text(spaced, encoding="utf-8")
    both_red = _night(c1, {"3.12": ({_A: tor.FAILED}, None), "3.13": ({_A: tor.FAILED}, None)},
                      run_id=101, created_at="2026-11-02T02:17:00Z")
    assert _publish(tmp_path, repo, both_red, None) == 0
    assert green_file.read_text(encoding="utf-8") == spaced


def test_an_incomplete_night_does_not_move_the_pointer(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, c1, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, day=1), None) == 0
    incomplete = _night(c1, {"3.12": ({_A: tor.PASSED}, None), "3.13": (None, None)}, run_id=101)
    assert incomplete["nightly"]["verdict"] == nc.INCOMPLETE
    assert _publish(tmp_path, repo, incomplete, None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c0


def test_an_unknown_ancestry_keeps_the_pointer(tmp_path: Path, main_repo: tuple[Path, list[str], str]) -> None:
    repo, (c0, _, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, day=1), None) == 0
    assert _publish(tmp_path, repo, _green("f" * 40, 101, day=2), None) == 1
    assert _last_green(tmp_path)["canary"]["sha"] == c0
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    assert nc.git_is_ancestor(not_a_repo, c0, c0) is None


def test_a_green_night_on_a_feature_commit_above_the_pointer_moves_nothing(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str],
) -> None:
    """The commit descends from the pointer, so forward-only alone would take it."""
    repo, (_, _, c2), _ = main_repo
    feature = _git(repo, "commit-tree", f"{c2}^{{tree}}", "-p", c2, "-m", "feature")
    assert _publish(tmp_path, repo, _green(c2, 100, day=1), None) == 0
    before = _last_green(tmp_path)
    assert _publish(tmp_path, repo, _green(feature, 101, day=2), None) == 1
    assert _last_green(tmp_path) == before
    assert "is not on main" in capsys.readouterr().out
    assert (tmp_path / "branch" / "runs" / "2026-11-02-101-a1" / "record.json").is_file()


@pytest.mark.parametrize("off_main", ["side", "unknown"])
def test_the_first_pointer_needs_a_commit_on_main_too(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], off_main: str,
) -> None:
    repo, (_, _, c2), side = main_repo
    sha = side if off_main == "side" else "f" * 40
    assert _publish(tmp_path, repo, _green(sha, 100, day=1), None) == 1
    assert _last_green(tmp_path) == {"canary": None, "legs": {}}
    assert _publish(tmp_path, repo, _green(c2, 101, day=2), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c2


def test_a_missing_main_repo_moves_nothing_exits_1_and_blames_no_commit(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, (c0, c1, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, day=1), None) == 0
    capsys.readouterr()
    assert _publish(tmp_path, tmp_path / "no-repo", _green(c1, 101, day=2), None) == 1
    assert _last_green(tmp_path)["canary"]["sha"] == c0
    out = capsys.readouterr().out
    assert "could not be checked against main" in out and "is not on main" not in out


def test_a_pointer_off_main_exits_1_each_night_until_a_green_night_repoints_it(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str],
) -> None:
    """No commit on main descends from it, so forward-only alone freezes it for good."""
    repo, (c0, c1, c2), side = main_repo
    branch = tmp_path / "branch"
    branch.mkdir()
    (branch / "runs" / "old").mkdir(parents=True)
    (branch / "runs" / "old" / "record.json").write_text("{}", encoding="utf-8")
    stale = {"sha": side, "path": "runs/old", "run_id": 1, "date": "2026-10-01"}
    frozen = {"canary": stale, "legs": {"3.12": stale}}
    (branch / "last-green.json").write_text(json.dumps(frozen), encoding="utf-8")
    assert _publish(tmp_path, repo, _red(c0, 100, day=1), None) == 1
    out = capsys.readouterr().out
    assert f"::error::last green canary: names {side[:12]}" in out
    assert f"::error::last green leg 3.12: names {side[:12]}" in out  # a red night changes no entry, green leg or not
    assert _last_green(tmp_path) == frozen
    assert _publish(tmp_path, repo, _red(c1, 101, day=2), None) == 1
    capsys.readouterr()
    assert _publish(tmp_path, repo, _green(c2, 102, day=3), None) == 0
    out = capsys.readouterr().out
    assert f"::warning::last green canary: {side[:12]} is not on main, re-pointed to {c2[:12]}" in out
    assert "::error::" not in out
    assert _last_green(tmp_path)["canary"]["sha"] == c2


def test_a_pointer_git_cannot_place_is_repaired_the_same_way(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (_, _, c2), _ = main_repo
    branch = tmp_path / "branch"
    branch.mkdir()
    stale = {"sha": "f" * 40, "path": "runs/old", "run_id": 1, "date": "2026-10-01"}
    (branch / "last-green.json").write_text(json.dumps({"canary": stale, "legs": {}}), encoding="utf-8")
    assert _publish(tmp_path, repo, _green(c2, 102, day=3), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c2


def _stale_clone(tmp_path: Path, full: Path, at: str) -> Path:
    """A repository whose main is `full`'s history up to `at`, so it has never
    fetched the commits after that."""
    _git(full, "branch", "--force", "up-to", at)
    stale = tmp_path / "stale"
    stale.mkdir()
    _git(stale, "init", "-q", "-b", "other")
    _git(stale, "fetch", "-q", str(full), "up-to:refs/heads/main")
    return stale


def test_a_stale_clone_cannot_move_a_newer_pointer_back(tmp_path: Path) -> None:
    """The pointer is a commit this clone never fetched, which says nothing about main."""
    full, (_, c1, c2), _ = _make_main_repo(tmp_path / "full")
    assert _publish(tmp_path, full, _green(c2, 10, day=2), None) == 0
    before = _last_green(tmp_path)
    stale = _stale_clone(tmp_path, full, c1)
    assert nc.git_is_ancestor(stale, c2, "main") is None
    assert _publish(tmp_path, stale, _green(c1, 9, day=1), None) == 1
    assert _last_green(tmp_path) == before
    assert _publish(tmp_path, stale, _red(c1, 11, day=3), None) == 1
    assert _last_green(tmp_path) == before


def test_a_shallow_clone_moves_a_pointer_only_on_a_newer_run(tmp_path: Path) -> None:
    full, (c0, _, c2), _ = _make_main_repo(tmp_path / "full")
    assert _publish(tmp_path, full, _green(c0, 5, day=1), None) == 0
    _git(full.parent, "clone", "-q", "--depth", "1", f"file://{full}", "shallow")
    shallow = full.parent / "shallow"
    assert nc.git_is_ancestor(shallow, c0, "origin/main") is None
    assert _publish(tmp_path, shallow, _green(c2, 4, day=2), None) == 1
    assert _last_green(tmp_path)["canary"]["sha"] == c0
    assert _publish(tmp_path, shallow, _green(c2, 6, day=3), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c2


def test_a_history_rewrite_is_recovered_by_a_newer_run_and_not_an_older_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    repo, (_, _, c2), side = _make_main_repo(tmp_path / "main")
    assert _publish(tmp_path, repo, _green(c2, 10, day=1), None) == 0
    _git(repo, "update-ref", "refs/heads/main", side)
    assert nc.git_is_ancestor(repo, c2, "main") is False
    capsys.readouterr()
    assert _publish(tmp_path, repo, _green(side, 9, day=2), None) == 1
    out = capsys.readouterr().out
    assert _last_green(tmp_path)["canary"]["sha"] == c2
    assert f"canary: refused {side[:12]}: run (9, 1) is not newer than run (10, 1)" in out
    assert _publish(tmp_path, repo, _green(side, 11, day=3), None) == 0
    out = capsys.readouterr().out
    assert _last_green(tmp_path)["canary"]["sha"] == side
    assert f"::warning::last green canary: {c2[:12]} is not on main, re-pointed to {side[:12]}" in out
    assert "::error::" not in out


@pytest.mark.parametrize("answer", [False, None], ids=["not-on-main", "cannot-place"])
@pytest.mark.parametrize(
    ("run", "moves"),
    [((10, 1), False), ((10, 2), True), ((9, 3), False), ((11, 1), True)],
    ids=["same-run-and-attempt", "newer-attempt", "older-run", "newer-run"],
)
def test_recovery_needs_a_newer_run_whichever_way_git_answers(
    answer: bool | None, run: tuple[int, int], moves: bool,
) -> None:
    old = {"canary": {"sha": SHA1, "path": "runs/a", "run_id": 10, "attempt": 1}, "legs": {}}
    new, _, problems = nc.next_last_green(
        old, _green(SHA2, run[0], run_attempt=run[1]), lambda a, b: True, lambda sha: True if sha == SHA2 else answer,
    )
    assert new["canary"]["sha"] == (SHA2 if moves else SHA1)
    assert any("refused" in p for p in problems) is not moves


@pytest.mark.parametrize("night", ["red", "incomplete"])
def test_a_night_that_is_not_green_changes_no_leg_entry_even_for_a_green_leg(night: str) -> None:
    """3.12 is green tonight, and a repair of its unplaceable entry waits for a green night."""
    old = {"canary": None, "legs": {"3.12": {"sha": SHA1, "path": "runs/a", "run_id": 1, "attempt": 1}}}
    record = (_red(SHA2, 2) if night == "red"
              else _night(SHA2, {"3.12": ({_A: tor.PASSED}, None), "3.13": (None, None)}, run_id=2))
    assert record["nightly"]["status"]["3.12"] == nc.GREEN
    new, notes, _ = nc.next_last_green(old, record, lambda a, b: True, lambda sha: None if sha == SHA1 else True)
    assert new == old and notes == []


def test_main_is_origin_main_before_a_local_main_and_the_flag_overrides_both(tmp_path: Path) -> None:
    repo, (c0, c1, c2), side = _make_main_repo(tmp_path / "main")
    _git(repo, "update-ref", "refs/remotes/origin/main", c1)
    assert nc.resolve_main_ref(repo, None) == "origin/main"
    assert _publish(tmp_path, repo, _green(c2, 100, day=1), None) == 1
    assert _last_green(tmp_path)["canary"] is None
    _git(repo, "update-ref", "refs/heads/other", side)
    assert _publish(tmp_path, repo, _green(side, 101, day=2), None, "--main-ref", "other") == 0
    assert _last_green(tmp_path)["canary"]["sha"] == side
    assert nc.resolve_main_ref(repo, "no-such-ref") is None
    assert _publish(tmp_path, repo, _green(c0, 102, day=3), None, "--main-ref", "no-such-ref") == 1
    assert _last_green(tmp_path)["canary"]["sha"] == side


def test_next_last_green_asks_about_each_entry_it_would_move() -> None:
    calls: list[tuple[str, str]] = []

    def is_ancestor(old: str, new: str) -> bool | None:
        calls.append((old, new))
        return True

    old = {"canary": {"sha": SHA1, "path": "runs/a", "run_id": 1, "date": "2026-11-03"}, "legs": {}}
    new, notes, problems = nc.next_last_green(old, _green(SHA2, 2), is_ancestor, lambda sha: True)
    assert new["canary"]["sha"] == SHA2 and set(new["legs"]) == {"3.12", "3.13"}
    assert calls == [(SHA1, SHA2)]
    assert notes[0] == f"last green canary: {SHA1[:12]} -> {SHA2[:12]}"
    assert problems == []


def test_ancestry_git_cannot_answer_keeps_the_pointer_and_is_a_problem() -> None:
    old = {"canary": {"sha": SHA1, "path": "runs/a", "run_id": 1}, "legs": {}}
    new, _, problems = nc.next_last_green(old, _green(SHA2, 2), lambda a, b: None, lambda sha: True)
    assert new["canary"]["sha"] == SHA1
    assert f"canary: stays at {SHA1[:12]}, ancestry against {SHA2[:12]} unknown" in problems


def test_a_missing_triage_still_publishes_the_record(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    assert _publish(tmp_path, repo, _red(c0, 100), None) == 0
    assert (tmp_path / "branch/runs/2026-11-04-100-a1/record.json").is_file()
    assert not (tmp_path / "branch/runs/2026-11-04-100-a1/triage.json").exists()
    assert _index(tmp_path)[0]["events"] == []
    assert "no triage" in capsys.readouterr().err


def test_publish_dry_run_writes_nothing(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100), None, "--dry-run") == 0
    assert list((tmp_path / "branch").iterdir()) == []
    printed = capsys.readouterr().out
    assert "would write runs/2026-11-04-100-a1/record.json" in printed
    assert "would write last-green.json" in printed


def test_publish_refuses_a_record_with_no_nightly_key(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    record = _green(c0, 100)
    record.pop("nightly")
    assert _publish(tmp_path, repo, record, None) == 2
    assert list((tmp_path / "branch").iterdir()) == []


def test_regressions_reads_the_last_green_night_publish_wrote(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, c1, _), _ = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, day=1), None) == 0
    current = tmp_path / "tonight.json"
    current.write_text(json.dumps(_red(c1, 101, day=2)), encoding="utf-8")
    out = tmp_path / "triage-out.json"
    root = _tree(tmp_path / "repo", "tests/test_a.py")
    args = ["regressions", "--current", str(current), "--last-green-dir", str(tmp_path / "branch"),
            "--canary-legs", "3.12,3.13", "--out", str(out)]
    assert nc.main([*args, "--repo-root", str(tmp_path / "no-such-dir")]) == 2
    assert nc.main([*args, "--repo-root", str(root)]) == 0
    triage = json.loads(out.read_text(encoding="utf-8"))
    assert triage["last_green"] == {"sha": c0, "path": "runs/2026-11-01-100-a1", "run_id": 100, "attempt": 1,
                                    "date": "2026-11-01"}
    node = "tests/test_a.py::test_one"
    assert triage["candidates"] == [{"test": _A, "node_id": node, "legs": ["3.13"], "before": {"3.13": tor.PASSED}}]
    assert triage["drift_check"] == {
        "sha": c0,
        "legs": {"3.13": {"tests": [_A], "node_ids": [node], "unmapped": [], "collection_errors": []}},
    }


# -------------------------------------------------------------- JSON shapes

def _with(key: str, value: object) -> dict[str, Any]:
    """A good red record with one `nightly` key replaced, or one top-level key
    for `sha` and `legs`."""
    record = _night(SHA1, {"3.12": ({_A: tor.FAILED}, None), "3.13": ({_A: tor.PASSED}, None)})
    if key in ("sha", "legs", "nightly"):
        record[key] = value
    else:
        record["nightly"][key] = value
    return record


def _green_with(key: str, value: object) -> dict[str, Any]:
    """A good green record with `legs` or one `nightly` key replaced."""
    record = _green(SHA1, 1)
    if key == "legs":
        record[key] = value
    else:
        record["nightly"][key] = value
    return record


_BAD_RECORDS: list[tuple[str, dict[str, Any]]] = [
    ("nightly-list", _with("nightly", [])),
    ("nightly-str", _with("nightly", "x")),
    ("sha-null", _with("sha", None)),
    ("sha-empty", _with("sha", "")),
    ("sha-not-hex", _with("sha", "--upload-pack=x")),
    ("legs-list", _with("legs", [])),
    ("leg-str", _with("legs", {"3.12": "x"})),
    ("leg-tests-list", _with("legs", {"3.12": {"tests": [], "counts": {}}})),
    ("leg-no-counts", _with("legs", {"3.12": {"tests": {}}})),
    ("status-list", _with("status", [])),
    ("status-unknown", _with("status", {"3.12": "fine"})),
    ("confirmed-list", _with("confirmed", [])),
    ("confirmed-str", _with("confirmed", {"3.12": "x"})),
    ("flaky-list", _with("flaky", [])),
    ("roles-list", _with("roles", [])),
    ("roles-unknown", _with("roles", {"3.12": "boss"})),
    ("mass-list", _with("mass_failure", [])),
    ("verdict-unknown", _with("verdict", "maybe")),
    ("run-id-str", _with("run_id", "x")),
    ("run-id-bool", _with("run_id", True)),
    ("run-id-0", _with("run_id", 0)),
    ("attempt-negative", _with("run_attempt", -1)),
    ("created-at-impossible", _with("created_at", "9999-99-99T00:00:00Z")),
    ("sha-too-short", _with("sha", "abc123")),
    ("attempt-null", _with("run_attempt", None)),
    ("created-at-null", _with("created_at", None)),
    ("created-at-word", _with("created_at", "Nov 4")),
    ("green-over-confirmed", _with("status", {"3.12": nc.GREEN, "3.13": nc.GREEN})),
    ("verdict-green-over-red-leg", _with("verdict", nc.GREEN)),
    ("green-no-legs", _green_with("legs", {})),
    ("green-null-canary-leg", _green_with("legs", {"3.12": None, "3.13": None})),
    ("green-no-canary-role", _green_with("roles", {"3.12": nc.EXTRA, "3.13": nc.EXTRA})),
    ("green-no-roles", _green_with("roles", {})),
]


@pytest.mark.parametrize("record", [r for _, r in _BAD_RECORDS], ids=[i for i, _ in _BAD_RECORDS])
def test_a_record_of_the_wrong_shape_is_a_usage_error_for_each_reader(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], record: dict[str, Any],
) -> None:
    """Exit 1 means a checked condition, and an uncaught exception exits 1."""
    repo, _, _ = main_repo
    current = tmp_path / "current.json"
    current.write_text(json.dumps(record), encoding="utf-8")
    assert nc.main(["regressions", "--current", str(current), "--last-green-dir", str(tmp_path / "none"),
                    "--out", str(tmp_path / "t.json")]) == 2
    assert _publish(tmp_path, repo, record, None) == 2
    assert list((tmp_path / "branch").iterdir()) == []


@pytest.mark.parametrize("record", [r for _, r in _BAD_RECORDS], ids=[i for i, _ in _BAD_RECORDS])
def test_a_baseline_of_the_wrong_shape_is_a_night_with_no_baseline(tmp_path: Path, record: dict[str, Any]) -> None:
    triage = _regress_on(tmp_path, _POINTS_AT_X, json.dumps(record).encode())
    assert triage["events"][0]["kind"] == "no-baseline" and "unreadable" in triage["events"][0]["reason"]


def test_the_records_this_script_writes_pass_its_own_check(tmp_path: Path) -> None:
    for record in (_green(SHA1, 1), _red(SHA1, 2), _night(SHA1, {"3.12": (None, None)})):
        nc.validate_record(json.loads(json.dumps(record)), "test")


_BAD_POINTERS = [
    ("array", "[]"), ("corrupt", "oops"), ("canary-str", '{"canary": "x"}'), ("canary-list", '{"canary": ["x"]}'),
    ("no-path", json.dumps({"canary": {"sha": SHA1, "run_id": 1}})),
    ("no-sha", json.dumps({"canary": {"path": "runs/x", "run_id": 1}})),
    ("no-run-id", json.dumps({"canary": {"sha": SHA1, "path": "runs/x"}})),
    ("attempt-str", json.dumps({"canary": {"sha": SHA1, "path": "runs/x", "run_id": 1, "attempt": "x"}})),
    ("legs-list", '{"canary": null, "legs": []}'),
    ("leg-str", '{"canary": null, "legs": {"3.12": "x"}}'),
]


@pytest.mark.parametrize("text", [t for _, t in _BAD_POINTERS], ids=[i for i, _ in _BAD_POINTERS])
def test_a_last_green_file_of_the_wrong_shape_is_a_night_with_no_baseline(tmp_path: Path, text: str) -> None:
    triage = _regress_on(tmp_path, text, None)
    assert triage["events"][0]["kind"] == "no-baseline"
    assert triage["events"][0]["reason"].startswith("last-green.json is unreadable")


@pytest.mark.parametrize("text", [t for _, t in _BAD_POINTERS], ids=[i for i, _ in _BAD_POINTERS])
def test_a_corrupt_last_green_file_still_publishes_the_night_and_exits_1(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str], text: str,
) -> None:
    """One bad file used to block every night's record until someone repaired it."""
    repo, (c0, _, _), _ = main_repo
    branch = tmp_path / "branch"
    branch.mkdir()
    (branch / "last-green.json").write_text(text, encoding="utf-8")
    triage = {"events": [{"key": "pr-1"}]}
    assert _publish(tmp_path, repo, _green(c0, 100), triage) == 1
    assert (branch / "last-green.json").read_text(encoding="utf-8") == text
    assert (branch / "runs/2026-11-04-100-a1/record.json").is_file()
    assert (branch / "runs/2026-11-04-100-a1/triage.json").is_file()
    assert _index(tmp_path)[0]["events"] == ["pr-1"]
    assert "::error::last-green.json is unreadable" in capsys.readouterr().out
    assert _publish(tmp_path, repo, _green(c0, 101, day=5), None, "--dry-run") == 1
    assert not (branch / "runs/2026-11-05-101-a1").exists()


def test_the_next_night_after_repairing_the_pointer_file_moves_it(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, c1, _), _ = main_repo
    (tmp_path / "branch").mkdir()
    (tmp_path / "branch" / "last-green.json").write_text("oops", encoding="utf-8")
    assert _publish(tmp_path, repo, _green(c0, 100), None) == 1
    (tmp_path / "branch" / "last-green.json").unlink()
    assert _publish(tmp_path, repo, _green(c1, 101, day=5), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c1


@pytest.mark.parametrize(
    "text",
    ["oops", "[]", '{"events": null}', '{"events": ["x"]}', '{"events": [{}]}', '{"events": [{"key": 1}]}'],
    ids=["corrupt", "array", "events-null", "event-str", "event-no-key", "key-not-str"],
)
def test_a_triage_file_of_the_wrong_shape_is_a_usage_error(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], text: str,
) -> None:
    repo, (c0, _, _), _ = main_repo
    branch = tmp_path / "branch"
    branch.mkdir()
    rec, tri = tmp_path / "record.json", tmp_path / "triage.json"
    rec.write_text(json.dumps(_red(c0, 100)), encoding="utf-8")
    tri.write_text(text, encoding="utf-8")
    assert nc.main(["publish", "--record", str(rec), "--triage", str(tri), "--branch-dir", str(branch),
                    "--main-repo", str(repo)]) == 2
    assert list(branch.iterdir()) == []


@pytest.mark.parametrize(
    "line", ["oops", "[]", '{"x": 1}', '{"run_id": "x", "run_attempt": 1}', '{"run_id": 1}'],
    ids=["corrupt", "array", "no-run", "run-str", "no-attempt"],
)
def test_a_corrupt_index_still_publishes_the_night_moves_the_pointer_and_exits_1(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], capsys: pytest.CaptureFixture[str], line: str,
) -> None:
    repo, (c0, _, _), _ = main_repo
    branch = tmp_path / "branch"
    branch.mkdir()
    (branch / "index.jsonl").write_text(line + "\n", encoding="utf-8")
    assert _publish(tmp_path, repo, _green(c0, 100), None) == 1
    assert (branch / "index.jsonl").read_text(encoding="utf-8") == line + "\n"
    assert (branch / "runs/2026-11-04-100-a1/record.json").is_file()
    assert _last_green(tmp_path)["canary"]["sha"] == c0
    assert "::error::index.jsonl is unreadable" in capsys.readouterr().out


def test_an_index_too_deep_to_parse_is_corrupt_rather_than_fatal(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, _, _), _ = main_repo
    (tmp_path / "branch").mkdir()
    (tmp_path / "branch" / "index.jsonl").write_text("[" * 200_000 + "\n", encoding="utf-8")
    assert _publish(tmp_path, repo, _red(c0, 100), None) == 1
    assert (tmp_path / "branch/runs/2026-11-04-100-a1/record.json").is_file()


@pytest.mark.parametrize(
    "exc",
    [OSError("x"), ValueError("x"), KeyError("x"), IndexError("x"), TypeError("x"), AttributeError("x"),
     RecursionError("x"), nc.ElementTree.ParseError("x")],
    ids=lambda e: type(e).__name__,
)
def test_main_turns_each_failure_it_can_meet_into_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exc: Exception,
) -> None:
    """An uncaught exception exits 1, which the docstring keeps for a checked condition."""
    def fail(args: Any) -> int:
        raise exc

    monkeypatch.setattr(nc, "cmd_env", fail)
    assert nc.main(["env", "--out", str(tmp_path / "e.json")]) == 2


# --------------------------------------------------------------------- env

def test_env_writes_the_interpreter_sqlite_image_and_uv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ImageOS", "ubuntu24")
    monkeypatch.setenv("ImageVersion", "20261102.1")
    monkeypatch.setattr(nc, "uv_version", lambda: "uv 0.9.0")
    out = tmp_path / "bundle" / "env.json"
    assert nc.main(["env", "--out", str(out)]) == 0
    env = json.loads(out.read_text(encoding="utf-8"))
    import sqlite3
    assert env == {"python": sys.version, "sqlite_version": sqlite3.sqlite_version, "ImageOS": "ubuntu24",
                   "ImageVersion": "20261102.1", "uv": "uv 0.9.0"}


def test_env_dry_run_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nc, "uv_version", lambda: None)
    out = tmp_path / "env.json"
    assert nc.main(["env", "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()


@pytest.mark.parametrize(
    "result",
    [FileNotFoundError("uv"), subprocess.TimeoutExpired(["uv"], 30),
     subprocess.CompletedProcess(["uv"], 2, stdout="", stderr="no")],
    ids=["absent", "hung", "failed"],
)
def test_uv_version_is_null_when_uv_does_not_answer(monkeypatch: pytest.MonkeyPatch, result: Any) -> None:
    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        assert kwargs["timeout"] > 0
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(nc.subprocess, "run", fake_run)
    assert nc.uv_version() is None


# ------------------------------------------------------------------ shape

def test_a_dry_run_leaves_no_bytecode_beside_the_script(tmp_path: Path) -> None:
    """Run as a process on a copy, with bytecode allowed, so the import of
    the sibling module is what would write `__pycache__`."""
    copy = tmp_path / "scripts"
    copy.mkdir()
    shutil.copy(_SCRIPT, copy)
    shutil.copy(_REPO / "scripts" / "test_outcome_record.py", copy)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
    r = subprocess.run(
        [sys.executable, str(copy / "nightly_canary.py"), "env", "--out", str(tmp_path / "env.json"), "--dry-run"],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in copy.iterdir()) == ["nightly_canary.py", "test_outcome_record.py"]


def test_importing_the_script_leaves_bytecode_writing_as_it_found_it() -> None:
    """The flag is process-wide; the test suite imports this module."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONDONTWRITEBYTECODE"}
    code = f"import sys; sys.path.insert(0, {str(_SCRIPT.parent)!r}); import nightly_canary; print(sys.dont_write_bytecode)"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert r.stdout.strip() == "False", r.stderr


def test_no_subcommand_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        nc.main([])
    assert exc.value.code == 2


def test_the_script_is_standard_library_only() -> None:
    """The publish job runs it on the runner's interpreter, without `uv sync`."""
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names - {"test_outcome_record"} <= set(sys.stdlib_module_names)


_IO_NAMES = frozenset({"print", "open", "input", "subprocess", "sys", "os", "Path"})
_IO_ATTRS = frozenset({"read_text", "write_text", "read_bytes", "write_bytes", "open", "run", "write",
                       "unlink", "mkdir", "iterdir", "is_file", "is_dir"})


def test_the_functions_the_docstring_calls_pure_do_no_io() -> None:
    assert nc.__doc__ is not None
    m = re.search(r"The functions above `(\w+)`\s+are pure", nc.__doc__)
    assert m, "the docstring no longer says which functions are pure"
    tree = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    names = [n.name for n in defs]
    assert m.group(1) in names
    impure: list[str] = []
    for node in defs[: names.index(m.group(1))]:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in _IO_NAMES:
                impure.append(f"{node.name}: {sub.id}")
            elif isinstance(sub, ast.Attribute) and sub.attr in _IO_ATTRS:
                impure.append(f"{node.name}: .{sub.attr}")
    assert impure == []
    assert len(defs[: names.index(m.group(1))]) > 15
