"""#1786 shape B: the nightly canary's record, triage input and results branch.

`scripts/nightly_canary.py` maps a leg's red tests to node ids for the
rerun, sorts them into flaky and confirmed, builds the night's format-1
record, compares the canary legs with the last green night and writes the
night to a checkout of the `test-results` branch. These tests run its pure
functions on synthetic JUnit reports in the shapes pytest 9 writes
(classname and name of a module test, a nested class, parameter ids with
dots, brackets, spaces and `::`, and a collection error with an empty
classname, each checked against a real run), and run its subcommands
against temporary files and a temporary git repository. Nothing here
calls the network or `gh`.
"""
from __future__ import annotations

import ast
import json
import os
import re
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
        ("tests.foo.test_bar::test_y", "tests/foo/test_bar.py::test_y"),
        ("tests.foo.TestF::test_z", "tests/foo.py::TestF::test_z"),
        ("::tests.test_broken", None),
        ("tests.gone::test_x", None),
        ("tests.test_a", None),
    ],
    ids=["module", "class", "nested-class", "param-dot", "param-space", "param-brackets",
         "param-colons", "param-escape", "longest-prefix-dir", "module-beside-dir",
         "collection-error", "deleted-module", "no-name"],
)
def test_a_junit_id_maps_to_the_node_id_pytest_reruns(tmp_path: Path, test: str, expected: str | None) -> None:
    """Each shape pytest 9 writes. A module that shares its name with a
    package resolves to the longer path first."""
    root = _tree(tmp_path, "tests/test_a.py", "tests/sub/deep/test_p.py", "tests/foo.py", "tests/foo/test_bar.py")
    assert nc.node_id(test, lambda rel: (root / rel).is_file()) == expected


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


@pytest.mark.parametrize(("n", "rerun", "ids"), [(50, "true", 50), (51, "false", 0)])
def test_more_than_fifty_red_tests_is_a_mass_failure_with_no_rerun(
    tmp_path: Path, n: int, rerun: str, ids: int,
) -> None:
    root = _tree(tmp_path / "repo", "tests/test_a.py")
    junit = tmp_path / "junit.xml"
    junit.write_text(_report(*(_case("tests.test_a", f"test_{i}", _FAIL) for i in range(n))), encoding="utf-8")
    out, gh_out = tmp_path / "ids.txt", tmp_path / "gh_output"
    assert nc.main(["failed-ids", "--junit", str(junit), "--repo-root", str(root), "--out", str(out),
                    "--github-output", str(gh_out)]) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == ids
    assert f"rerun={rerun}\n" in gh_out.read_text(encoding="utf-8")
    assert f"mass_failure={'true' if n > 50 else 'false'}\n" in gh_out.read_text(encoding="utf-8")


def test_a_run_whose_only_red_test_maps_to_nothing_is_not_rerun(tmp_path: Path) -> None:
    """An empty `@file` would make pytest run the whole suite."""
    plan = nc.plan_rerun({"::tests.test_gone": tor.ERROR}, lambda rel: False, 50)
    assert plan == {"count": 1, "rerun": False, "mass_failure": False, "ids": [], "unmapped": ["::tests.test_gone"]}


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
        (1, {_A: tor.PASSED}, None, 1, nc.BROKEN, [], []),
        (1, {_A: tor.FAILED}, {_A: tor.PASSED}, 0, nc.GREEN, [_A], []),
    ],
    ids=["clean", "flaky", "confirmed", "no-rerun", "absent-from-rerun", "skipped-on-rerun",
         "flaky-and-confirmed", "collection-error", "interrupted", "no-tests", "rc1-nothing-red",
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
        argv += ["--rerun", str(tmp_path / "rerun.xml")]
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


def test_verdict_dry_run_writes_nothing_and_keeps_the_exit_status(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(_junit({_A: tor.FAILED}), encoding="utf-8")
    out = tmp_path / "verdict.json"
    assert nc.main(["verdict", "--rc", "1", "--junit", str(junit), "--out", str(out), "--dry-run"]) == 1
    assert not out.exists()


def test_verdict_on_an_unreadable_report_is_a_usage_error(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    assert nc.main(["verdict", "--rc", "1", "--junit", str(junit)]) == 2
    junit.write_text(_junit({_A: tor.FAILED}), encoding="utf-8")
    assert nc.main(["verdict", "--rc", "1", "--junit", str(junit), "--rerun", str(tmp_path / "none.xml")]) == 2


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
    rc, record = _record_cmd(tmp_path, f"3.12={root}", f"3.1={root}")
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


def test_a_broken_canary_leg_makes_the_night_red(tmp_path: Path) -> None:
    _bundle(tmp_path / "3.12", {_A: tor.PASSED})
    _bundle(tmp_path / "3.13", {_A: tor.PASSED}, rc=2)
    rc, record = _record_cmd(tmp_path, f"3.12={tmp_path / '3.12'}", f"3.13={tmp_path / '3.13'}")
    assert rc == 0 and record is not None
    assert record["nightly"]["status"]["3.13"] == nc.BROKEN
    assert record["nightly"]["verdict"] == nc.RED


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


def test_record_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _bundle(tmp_path / "b", {_A: tor.FAILED})
    rc, record = _record_cmd(tmp_path, f"3.12={tmp_path / 'b'}", extra=["--dry-run"])
    assert rc == 0 and record is None
    assert "night: red" in capsys.readouterr().out


# ------------------------------------------------------------ regressions

_LEGS = ["3.12", "3.13"]


def test_a_flaky_failure_on_the_last_green_night_counts_as_passed() -> None:
    green = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.FAILED}, {_A: tor.PASSED})})
    assert green["nightly"]["verdict"] == nc.GREEN
    tonight = _night(SHA2, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.FAILED}, {_A: tor.FAILED})})
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["candidates"] == [{"test": _A, "legs": ["3.13"], "before": {"3.13": tor.PASSED}}]
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
    assert triage["new_failing"] == [{"test": _C, "legs": ["3.12", "3.13"]}]
    assert [c["test"] for c in triage["candidates"]] == [_A]
    assert triage["drift_check"] == {"sha": SHA1, "python": "3.12", "tests": [_A]}


def test_a_test_skipped_on_the_last_green_night_that_now_fails_is_a_candidate() -> None:
    green = _night(SHA1, {"3.12": ({_A: tor.SKIPPED}, None), "3.13": ({_A: tor.PASSED}, None)})
    tonight = _night(SHA2, {"3.12": ({_A: tor.FAILED}, None), "3.13": ({_A: tor.PASSED}, None)})
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["candidates"] == [{"test": _A, "legs": ["3.12"], "before": {"3.12": tor.SKIPPED}}]


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
    assert triage["events"] == [{"key": f"no-baseline-{SHA2[:12]}", "kind": "no-baseline",
                                 "legs": ["3.12", "3.13"], "tests": [_A, _C]}]
    assert triage["candidates"] == [] and triage["drift_check"] is None


def test_a_red_night_on_the_last_green_commit_is_drift() -> None:
    green = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.PASSED}, None)})
    tonight = _night(SHA1, {"3.12": ({_A: tor.PASSED}, None), "3.13": ({_A: tor.FAILED}, None)}, run_id=101)
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["events"] == [{"key": f"drift-{SHA1[:12]}", "kind": "drift", "legs": ["3.13"], "tests": [_A]}]
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
    assert triage["events"] == [{"key": "canary-broken", "kind": "canary-broken", "legs": ["3.13"], "tests": []}]


def test_a_mass_failure_drift_checks_an_even_sample_of_twenty() -> None:
    tests = {f"tests.test_a::test_{i:02}": tor.PASSED for i in range(60)}
    green = _night(SHA1, {"3.12": (tests, None), "3.13": (tests, None)})
    failing = {t: tor.FAILED for t in tests}
    tonight = _night(SHA2, {"3.12": (failing, None), "3.13": (failing, None)})
    assert tonight["nightly"]["mass_failure"] == {"3.12": True, "3.13": True}
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["mass_failure"] is True
    assert len(triage["candidates"]) == 60
    check = triage["drift_check"]
    assert check["python"] == "3.13"
    assert check["tests"] == [f"tests.test_a::test_{i:02}" for i in range(0, 60, 3)]


def test_below_the_threshold_every_candidate_is_drift_checked() -> None:
    tests = {f"tests.test_a::test_{i:02}": tor.PASSED for i in range(30)}
    green = _night(SHA1, {"3.12": (tests, None), "3.13": (tests, None)})
    tonight = _night(SHA2, {"3.12": (tests, None), "3.13": ({t: tor.FAILED for t in tests}, None)})
    triage = nc.find_regressions(tonight, green, {"sha": SHA1, "path": "runs/x"}, _LEGS)
    assert triage["mass_failure"] is False
    assert triage["drift_check"]["tests"] == sorted(tests)


def test_regressions_dry_run_writes_nothing(tmp_path: Path) -> None:
    current = tmp_path / "record.json"
    current.write_text(json.dumps(_night(SHA2, {"3.12": ({_A: tor.FAILED}, None)})), encoding="utf-8")
    out = tmp_path / "triage.json"
    assert nc.main(["regressions", "--current", str(current), "--last-green-dir", str(tmp_path / "branch"),
                    "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()


def test_regressions_on_a_broken_results_branch_is_a_usage_error(tmp_path: Path) -> None:
    current = tmp_path / "record.json"
    current.write_text(json.dumps(_night(SHA2, {"3.12": ({_A: tor.FAILED}, None)})), encoding="utf-8")
    branch = tmp_path / "branch"
    branch.mkdir()
    (branch / "last-green.json").write_text(json.dumps({"canary": {"sha": SHA1, "path": "runs/gone"}}),
                                            encoding="utf-8")
    assert nc.main(["regressions", "--current", str(current), "--last-green-dir", str(branch),
                    "--out", str(tmp_path / "t.json")]) == 2


# ----------------------------------------------------------------- publish

def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "HOME": str(repo.parent), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    r = subprocess.run(
        ["git", "-c", "user.name=canary", "-c", "user.email=canary@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=repo, env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    return r.stdout.strip()


@pytest.fixture
def main_repo(tmp_path: Path) -> tuple[Path, list[str], str]:
    """A main branch c0 -> c1 -> c2, and a commit off c0 that is not on it."""
    repo = tmp_path / "main"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    shas = []
    for i in range(3):
        _git(repo, "commit", "-q", "--allow-empty", "-m", f"c{i}")
        shas.append(_git(repo, "rev-parse", "HEAD"))
    side = _git(repo, "commit-tree", f"{shas[0]}^{{tree}}", "-p", shas[0], "-m", "side")
    return repo, shas, side


def _publish(tmp_path: Path, repo: Path, record: dict[str, Any], triage: dict[str, Any] | None,
             *extra: str) -> int:
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
    pointer = {"sha": shas[0], "path": "runs/2026-11-04-100-a1", "run_id": 100, "date": "2026-11-04"}
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
    assert [(e["run_id"], e["run_attempt"]) for e in _index(tmp_path)] == [(101, 1), (100, 3)]
    assert _index(tmp_path)[1]["path"] == "runs/2026-11-04-100-a3"
    # An older attempt published late leaves the newest attempt's line.
    assert _publish(tmp_path, repo, _red(shas[1], 100, run_attempt=2), None) == 0
    assert [(e["run_id"], e["run_attempt"]) for e in _index(tmp_path)] == [(101, 1), (100, 3)]


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


def test_the_last_green_pointer_moves_only_on_green_and_only_forward(
    tmp_path: Path, main_repo: tuple[Path, list[str], str],
) -> None:
    repo, (c0, c1, c2), side = main_repo
    assert _publish(tmp_path, repo, _green(c0, 100, day=1), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c0
    assert _publish(tmp_path, repo, _red(c1, 101, day=2), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c0
    # The red night's 3.12 leg was green, so that leg's pointer moved.
    assert _last_green(tmp_path)["legs"]["3.12"]["sha"] == c1
    assert _last_green(tmp_path)["legs"]["3.13"]["sha"] == c0
    assert _publish(tmp_path, repo, _green(c2, 102, day=3), None) == 0
    assert _last_green(tmp_path)["canary"] == {"sha": c2, "path": "runs/2026-11-03-102-a1", "run_id": 102,
                                               "date": "2026-11-03"}
    assert _publish(tmp_path, repo, _green(c1, 103, day=4), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c2
    assert _publish(tmp_path, repo, _green(side, 104, day=5), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c2
    assert _publish(tmp_path, repo, _green(c2, 105, day=6), None) == 0
    assert _last_green(tmp_path)["canary"]["run_id"] == 105


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
    assert _publish(tmp_path, repo, _green("f" * 40, 101, day=2), None) == 0
    assert _last_green(tmp_path)["canary"]["sha"] == c0
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    assert nc.git_is_ancestor(not_a_repo, c0, c0) is None


def test_next_last_green_asks_about_each_entry_it_would_move() -> None:
    calls: list[tuple[str, str]] = []

    def is_ancestor(old: str, new: str) -> bool | None:
        calls.append((old, new))
        return True

    old = {"canary": {"sha": SHA1, "path": "runs/a", "run_id": 1, "date": "2026-11-03"}, "legs": {}}
    new, notes = nc.next_last_green(old, _green(SHA2, 2), is_ancestor)
    assert new["canary"]["sha"] == SHA2 and set(new["legs"]) == {"3.12", "3.13"}
    assert calls == [(SHA1, SHA2)]
    assert notes[0] == f"canary: {SHA1[:12]} -> {SHA2[:12]}"


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


@pytest.mark.parametrize(
    "mutate",
    [lambda r: r.pop("nightly"), lambda r: r["nightly"].update(created_at="Nov 4")],
    ids=["no-nightly-key", "bad-date"],
)
def test_publish_refuses_a_record_it_cannot_place(
    tmp_path: Path, main_repo: tuple[Path, list[str], str], mutate: Any,
) -> None:
    repo, (c0, _, _), _ = main_repo
    record = _green(c0, 100)
    mutate(record)
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
    assert nc.main(["regressions", "--current", str(current), "--last-green-dir", str(tmp_path / "branch"),
                    "--canary-legs", "3.12,3.13", "--out", str(out)]) == 0
    triage = json.loads(out.read_text(encoding="utf-8"))
    assert triage["last_green"] == {"sha": c0, "path": "runs/2026-11-01-100-a1", "run_id": 100, "date": "2026-11-01"}
    assert triage["candidates"] == [{"test": _A, "legs": ["3.13"], "before": {"3.13": tor.PASSED}}]
    assert triage["drift_check"] == {"sha": c0, "python": "3.13", "tests": [_A]}


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
