"""#1786 shape A: the per-PR test outcome record and its ratchet.

`scripts/test_outcome_record.py` reads the JUnit reports the `pytest`
legs upload, writes one record of every test's outcome, finds the record
of the last green run on main and diffs the two. These tests run its pure
functions on synthetic JUnit XML, in the shapes pytest writes (checked
against a pytest 9 run with a failing, erroring, skipping, xfailing and
strict-xpassing test, each also with a fixture that raises in teardown),
drive the baseline walk with a fake API, and pin
the `test-outcomes` job in ci.yml as text, the way the neighboring
workflow tests do.

The ruling of 2026-10-21 makes the skip/xfail ratchet advisory first and
blocking later. The workflow tests here pin the advisory wiring: no
`--block`, job-level `continue-on-error`, and a job name outside the
merge train's floor.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[1]
_CI = _REPO / ".github" / "workflows" / "ci.yml"
_ALLOWLIST = _REPO / ".github" / "test-outcome-allowlist.txt"
sys.path.insert(0, str(_REPO / "scripts"))

import test_outcome_record as tor  # noqa: E402
from merge_train_gate import FLOOR_NAMES  # noqa: E402

_DOWNLOAD_SHA = "d3f86a106a0bac45b974a628896c90dbdf5c8093"
_UPLOAD_SHA = "ea165f8d65b6e75b540449e92b4886f43607fa02"


# ----------------------------------------------------------- fixtures

def _case(classname: str, name: str, body: str = "") -> str:
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
_XFAIL = '<skipped type="pytest.xfail" message="known bug" />'
_XPASS_STRICT = '<failure message="[XPASS(strict)] known bug">[XPASS(strict)] known bug</failure>'
_XPASS_LEGACY = '<skipped message="xfail-marked test passes unexpectedly" />'
_SKIP_PLAIN = '<skipped type="pytest.skip" message="needs sqlite 3.45">t.py:3: needs sqlite 3.45</skipped>'
_SKIP_PERF = (
    '<skipped type="pytest.skip" message="perf-gated: pass --run-perf to run">t.py:3: perf</skipped>'
)
_SKIP_BENCH = (
    '<skipped type="pytest.skip" message="AELFRICE_CORPUS_ROOT not set or not a directory; '
    'skipping bench-gate test (lab corpus absent by design)">t.py:3: corpus</skipped>'
)
_SKIP_MODULE = (
    '<skipped type="pytest.skip" message="corpus module \'retrieve\' missing under /x">t.py:3: m</skipped>'
)

_MIXED = _report(
    _case("tests.test_a", "test_pass"),
    _case("tests.test_a", "test_fail", _FAIL),
    _case("tests.test_a", "test_error", _ERROR),
    _case("tests.test_a", "test_xfail", _XFAIL),
    _case("tests.test_a", "test_xpass", _XPASS_STRICT),
    _case("tests.test_a", "test_skip", _SKIP_PLAIN),
    _case("tests.test_a", "test_perf", _SKIP_PERF),
    _case("tests.bench_gate.test_b", "test_gate", _SKIP_BENCH),
    _case("tests.test_a.TestGroup", "test_in_class"),
    _case("tests.test_a", "test_param[a b]"),
)


# ------------------------------------------------------- classification

@pytest.mark.parametrize(
    ("body", "outcome"),
    [
        ("", tor.PASSED),
        (_FAIL, tor.FAILED),
        (_ERROR, tor.ERROR),
        (_XFAIL, tor.XFAILED),
        (_XPASS_STRICT, tor.XPASSED),
        (_XPASS_LEGACY, tor.XPASSED),
        (_SKIP_PLAIN, tor.SKIPPED),
        (_SKIP_PERF, tor.NEVER_EXECUTED),
        (_SKIP_BENCH, tor.NEVER_EXECUTED),
        (_SKIP_MODULE, tor.NEVER_EXECUTED),
    ],
    ids=["pass", "fail", "error", "xfail", "xpass-strict", "xpass-legacy",
         "skip", "perf", "bench-gate", "corpus-module"],
)
def test_each_junit_shape_maps_to_one_outcome(body: str, outcome: str) -> None:
    outcomes = tor.parse_junit(_report(_case("tests.test_a", "test_x", body)))
    assert outcomes == {"tests.test_a::test_x": outcome}


def test_every_classified_outcome_is_a_declared_one() -> None:
    for outcome in tor.parse_junit(_MIXED).values():
        assert outcome in tor.OUTCOMES


_TEARDOWN_ERROR = '<error message="failed on teardown with &quot;RuntimeError: boom&quot;">boom</error>'


@pytest.mark.parametrize(
    ("entries", "outcome"),
    [
        ([_FAIL, _TEARDOWN_ERROR], tor.FAILED),
        ([_TEARDOWN_ERROR, _FAIL], tor.FAILED),
        ([_TEARDOWN_ERROR], tor.ERROR),
        ([_SKIP_PLAIN + _TEARDOWN_ERROR], tor.ERROR),
        ([_XFAIL + _XFAIL], tor.XFAILED),
        ([_XPASS_STRICT + _XFAIL], tor.XPASSED),
    ],
    ids=["fail-two-entries", "fail-two-entries-reversed", "pass", "skip", "xfail", "xpass-strict"],
)
def test_a_teardown_error_in_each_shape_pytest_writes(entries: list[str], outcome: str) -> None:
    """pytest 9 writes a failed call and its teardown error as two
    `<testcase>` entries with one id, and every other call outcome with its
    teardown error as one entry. A failure outranks the error in either
    order; an error outranks a skip."""
    report = _report(*(_case("tests.test_a", "test_x", body) for body in entries))
    outcomes = tor.parse_junit(report)
    assert outcomes == {"tests.test_a::test_x": outcome}
    assert tor.count_outcomes(outcomes)["total"] == 1


@pytest.mark.parametrize("reason", [
    "perf test gated on --run-perf",
    "perf benchmark; pass --run-perf to enable",
    "perf-gated: pass --run-perf to run",
])
def test_every_perf_skip_wording_in_the_suite_is_never_executed(reason: str) -> None:
    """The four latency tests word the reason differently; all name the flag."""
    assert tor.is_never_executed_reason(reason)


@pytest.mark.parametrize("reason", [
    "needs sqlite 3.45",
    "requires the 'archive' extra",
    "perf",
    "bench-gate",
])
def test_an_ordinary_skip_reason_is_not_never_executed(reason: str) -> None:
    assert not tor.is_never_executed_reason(reason)


def test_the_test_id_is_classname_and_name() -> None:
    outcomes = tor.parse_junit(_report(
        _case("tests.test_a.TestGroup", "test_in_class"),
        _case("tests.test_a", "test_param[c::d]"),
        _case("tests.test_b", "test_param[c::d]", _FAIL),
    ))
    assert outcomes == {
        "tests.test_a.TestGroup::test_in_class": tor.PASSED,
        "tests.test_a::test_param[c::d]": tor.PASSED,
        "tests.test_b::test_param[c::d]": tor.FAILED,
    }


def test_counts_cover_every_outcome_and_the_total() -> None:
    counts = tor.count_outcomes(tor.parse_junit(_MIXED))
    assert counts == {
        tor.PASSED: 3, tor.FAILED: 1, tor.ERROR: 1, tor.SKIPPED: 1, tor.XFAILED: 1,
        tor.XPASSED: 1, tor.NEVER_EXECUTED: 2, "total": 10,
    }


def test_a_malformed_report_raises_rather_than_recording_zero_tests() -> None:
    with pytest.raises(Exception):
        tor.parse_junit("<testsuites><testsuite>")


# --------------------------------------------------------------- record

def test_the_record_holds_each_leg_sorted_and_a_missing_leg_as_null() -> None:
    record = tor.build_record({"3.13": _MIXED, "3.12": None}, sha="abc123")
    assert record["format"] == tor.RECORD_FORMAT
    assert record["sha"] == "abc123"
    assert list(record["legs"]) == ["3.12", "3.13"]
    assert record["legs"]["3.12"] is None
    leg = record["legs"]["3.13"]
    assert list(leg["tests"]) == sorted(leg["tests"])
    assert leg["counts"]["total"] == 10
    assert json.loads(json.dumps(record)) == record


# ----------------------------------------------------------------- diff

@pytest.mark.parametrize("after", sorted(tor.RATCHETED_OUTCOMES))
@pytest.mark.parametrize("before", [None, *tor.OUTCOMES])
def test_moving_to_a_ratcheted_outcome_is_a_violation_unless_unchanged(
    before: str | None, after: str,
) -> None:
    assert tor.is_violation(before, after) is (before != after)


@pytest.mark.parametrize("after", [None, tor.PASSED, tor.FAILED, tor.ERROR, tor.XPASSED])
@pytest.mark.parametrize("before", [None, *tor.OUTCOMES])
def test_moving_anywhere_else_is_not_a_violation(before: str | None, after: str | None) -> None:
    assert tor.is_violation(before, after) is False


def test_the_ratcheted_set_is_skip_xfail_and_never_executed() -> None:
    assert tor.RATCHETED_OUTCOMES == {tor.SKIPPED, tor.XFAILED, tor.NEVER_EXECUTED}


def test_diff_leg_sorts_transitions_into_violations_recovered_red_added_removed() -> None:
    baseline = {
        "t::stays": tor.PASSED,
        "t::now_skips": tor.PASSED,
        "t::now_xfails": tor.PASSED,
        "t::recovers": tor.SKIPPED,
        "t::still_skips": tor.SKIPPED,
        "t::removed": tor.PASSED,
        "t::now_fails": tor.PASSED,
    }
    current = {
        "t::stays": tor.PASSED,
        "t::now_skips": tor.SKIPPED,
        "t::now_xfails": tor.XFAILED,
        "t::recovers": tor.PASSED,
        "t::still_skips": tor.SKIPPED,
        "t::now_fails": tor.FAILED,
        "t::new_pass": tor.PASSED,
        "t::new_perf": tor.NEVER_EXECUTED,
        "t::new_error": tor.ERROR,
        "t::new_xpass": tor.XPASSED,
    }
    d = tor.diff_leg("3.13", current, baseline)
    assert [(t.test, t.before, t.after) for t in d.violations] == [
        ("t::new_perf", None, tor.NEVER_EXECUTED),
        ("t::now_skips", tor.PASSED, tor.SKIPPED),
        ("t::now_xfails", tor.PASSED, tor.XFAILED),
    ]
    assert [t.test for t in d.recovered] == ["t::recovers"]
    assert [(t.test, t.after) for t in d.red] == [
        ("t::new_error", tor.ERROR),
        ("t::new_xpass", tor.XPASSED),
        ("t::now_fails", tor.FAILED),
    ]
    assert (d.added, d.removed) == (4, 1)
    assert d.baseline_present and d.notice is None


@pytest.mark.parametrize("after", [*tor.OUTCOMES, None])
@pytest.mark.parametrize("before", tor.OUTCOMES)
def test_only_a_ratcheted_outcome_that_now_passes_is_recovered(before: str, after: str | None) -> None:
    """A skipped test that now fails has not recovered; the red list names it."""
    current = {} if after is None else {"t::x": after}
    d = tor.diff_leg("3.13", current, {"t::x": before})
    recovered = [(t.test, t.before, t.after) for t in d.recovered]
    expected = before in tor.RATCHETED_OUTCOMES and after == tor.PASSED
    assert recovered == ([("t::x", before, after)] if expected else [])


def test_the_allowlist_exempts_by_glob_and_ignores_comments() -> None:
    text = "# windows-only tests skip on the linux legs\n\ntests.test_win*::*\n"
    allow = tor.parse_allowlist(text)
    assert allow == ["tests.test_win*::*"]
    assert tor.is_allowlisted("tests.test_win_paths::test_drive", allow)
    assert not tor.is_allowlisted("tests.test_posix::test_drive", allow)
    d = tor.diff_leg(
        "3.12",
        {"tests.test_win_paths::test_drive": tor.SKIPPED, "tests.test_posix::test_x": tor.SKIPPED},
        {"tests.test_win_paths::test_drive": tor.PASSED, "tests.test_posix::test_x": tor.PASSED},
        allow,
    )
    assert [t.test for t in d.violations] == ["tests.test_posix::test_x"]


def test_every_checked_in_allowlist_entry_is_one_test_id_glob() -> None:
    """A trailing comment or a bare module name would parse and match nothing."""
    for entry in tor.parse_allowlist(_ALLOWLIST.read_text(encoding="utf-8")):
        assert "::" in entry and not re.search(r"\s", entry), entry


def test_a_leg_without_a_baseline_reports_a_notice_and_no_violations() -> None:
    d = tor.diff_leg("3.12", {"t::a": tor.SKIPPED, "t::b": tor.FAILED}, None)
    assert d.violations == [] and not d.baseline_present
    assert d.notice is not None and "no baseline" in d.notice
    assert [t.test for t in d.red] == ["t::b"]


def _record(**legs: dict[str, str] | None) -> dict[str, Any]:
    return {
        "format": tor.RECORD_FORMAT,
        "sha": "base",
        "legs": {
            leg: None if tests is None else {"tests": tests, "counts": tor.count_outcomes(tests)}
            for leg, tests in legs.items()
        },
    }


def test_diff_records_walks_the_current_legs_and_tolerates_a_null_leg() -> None:
    current = {"format": 1, "sha": "head", "legs": {
        "3.13": {"tests": {"t::a": tor.SKIPPED}, "counts": {}},
        "3.12": None,
    }}
    diffs = tor.diff_records(current, _record(**{"3.13": {"t::a": tor.PASSED}}))
    assert [d.leg for d in diffs] == ["3.12", "3.13"]
    assert diffs[0].notice is not None and "no JUnit report" in diffs[0].notice
    assert [t.test for t in diffs[1].violations] == ["t::a"]
    assert tor.has_violations(diffs)


def test_diff_records_with_no_baseline_has_no_violations() -> None:
    current = _record(**{"3.13": {"t::a": tor.SKIPPED}})
    diffs = tor.diff_records(current, None)
    assert not tor.has_violations(diffs)
    assert all(not d.baseline_present for d in diffs)


def test_the_summary_names_the_mode_the_baseline_and_each_violation() -> None:
    current = _record(**{"3.13": {"t::a": tor.SKIPPED, "t::b": tor.PASSED}})
    baseline = _record(**{"3.13": {"t::a": tor.PASSED, "t::b": tor.SKIPPED}})
    diffs = tor.diff_records(current, baseline)
    text = tor.render_summary(diffs, baseline, None, block=False)
    assert text.startswith("## Test outcomes (advisory)\n")
    assert "Baseline: main @ `base`." in text
    assert "| 3.13 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 2 |" in text
    assert "| `t::a` | passed -> skipped |" in text
    assert "- `t::b`: skipped -> passed" in text
    assert "1 ratchet violation(s), 1 recovered, 0 added, 0 removed." in text
    assert "## Test outcomes (blocking)" in tor.render_summary(diffs, baseline, None, block=True)


def test_the_summary_without_a_baseline_carries_the_notice() -> None:
    current = _record(**{"3.13": {"t::a": tor.SKIPPED}})
    diffs = tor.diff_records(current, None)
    text = tor.render_summary(diffs, None, "artifacts expire after 30 days", block=False)
    assert "No baseline: artifacts expire after 30 days. The ratchet was not evaluated." in text
    assert "violation" not in text


def test_annotations_are_warnings_when_advisory_and_errors_when_blocking() -> None:
    current = _record(**{"3.13": {"t::a": tor.SKIPPED}})
    baseline = _record(**{"3.13": {"t::a": tor.PASSED}})
    diffs = tor.diff_records(current, baseline)
    assert tor.render_annotations(diffs, None, block=False) == [
        "::warning::test outcome ratchet, leg 3.13: t::a passed -> skipped",
    ]
    assert tor.render_annotations(diffs, None, block=True)[0].startswith("::error::")
    assert tor.render_annotations([], "gone", block=False) == [
        "::notice::test outcomes: no baseline: gone",
    ]


# ------------------------------------------------------------- baseline

class _FakeApi:
    """Commits on main, newest first, with their runs and artifacts."""

    def __init__(self, commits: list[str], runs: dict[str, list[dict[str, Any]]],
                 artifacts: dict[int, list[dict[str, Any]]]) -> None:
        self.commits = commits
        self.runs = runs
        self.artifacts = artifacts
        self.paths: list[str] = []

    def json(self, path: str) -> Any:
        self.paths.append(path)
        if "/commits?" in path:
            return [{"sha": sha} for sha in self.commits]
        m = re.search(r"head_sha=([^&]+)&status=success", path)
        if m:
            return {"workflow_runs": self.runs.get(m.group(1), [])}
        m = re.search(r"/runs/(\d+)/artifacts", path)
        assert m, path
        return {"artifacts": self.artifacts.get(int(m.group(1)), [])}

    def bytes(self, path: str) -> bytes:
        self.paths.append(path)
        import io
        import zipfile
        m = re.search(r"/artifacts/(\d+)/zip", path)
        assert m, path
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(f"junit-{m.group(1)}.xml", _report(_case("t", f"test_{m.group(1)}")))
        return buf.getvalue()


def _artifact(id_: int, name: str, expired: bool = False) -> dict[str, Any]:
    return {"id": id_, "name": name, "expired": expired}


def _walk(api: _FakeApi) -> tor.BaselineRef | None:
    return tor.choose_baseline(
        api.commits,
        lambda s: api.json(f"?head_sha={s}&status=success")["workflow_runs"],
        lambda r: api.json(f"/runs/{r}/artifacts")["artifacts"],
        ["3.12", "3.13"],
    )


def test_artifact_names_match_on_the_leg_prefix_and_ignore_the_merge_sha() -> None:
    legs = ["3.12", "3.13"]
    assert tor.artifact_leg("junit-3.13-2c5f46cb0000", legs) == "3.13"
    assert tor.artifact_leg("junit-3.12-1ed7e421", legs) == "3.12"
    assert tor.artifact_leg("junit-3.14-1ed7e421", legs) is None
    assert tor.artifact_leg("mutation-plan", legs) is None
    # The leg is a prefix match on the whole name, not a substring.
    assert tor.artifact_leg("test-outcome-record-junit-3.12-m1", legs) is None
    assert tor.artifact_leg("junit-3.120-m1", legs) is None


def test_the_walk_takes_the_base_commit_when_its_run_is_green_with_reports() -> None:
    api = _FakeApi(
        ["base", "older"],
        {"base": [{"id": 7}], "older": [{"id": 5}]},
        {7: [_artifact(70, "junit-3.12-m1"), _artifact(71, "junit-3.13-m1")]},
    )
    ref = _walk(api)
    assert ref == tor.BaselineRef(sha="base", run_id=7, artifacts={"3.12": 70, "3.13": 71})


def test_the_walk_skips_commits_without_a_run_and_runs_without_reports() -> None:
    """An inner commit of a merged PR has no run; a docs-only PR's run has no report."""
    api = _FakeApi(
        ["inner", "docs_only", "green"],
        {"docs_only": [{"id": 8}], "green": [{"id": 5}]},
        {8: [], 5: [_artifact(50, "junit-3.13-m0")]},
    )
    ref = _walk(api)
    assert ref == tor.BaselineRef(sha="green", run_id=5, artifacts={"3.13": 50})


def test_the_walk_skips_expired_artifacts_and_runs_out_to_none() -> None:
    api = _FakeApi(
        ["base"],
        {"base": [{"id": 7}]},
        {7: [_artifact(70, "junit-3.12-m1", expired=True)]},
    )
    ref = _walk(api)
    assert ref is None


def test_fetch_baseline_queries_by_head_sha_and_builds_the_record() -> None:
    api = _FakeApi(
        ["base", "green"],
        {"green": [{"id": 5}]},
        {5: [_artifact(50, "junit-3.12-m0"), _artifact(51, "junit-3.13-m0")]},
    )
    record = tor.fetch_baseline(api, "o/r", "base", "ci.yml", ["3.12", "3.13"], 30)  # type: ignore[arg-type]
    assert record["sha"] == "green" and record["run_id"] == 5
    assert record["legs"]["3.12"]["tests"] == {"t::test_50": tor.PASSED}
    assert record["legs"]["3.13"]["tests"] == {"t::test_51": tor.PASSED}
    assert tor.is_baseline_record(record)
    assert api.paths[0] == "/repos/o/r/commits?sha=base&per_page=30"
    assert "/repos/o/r/actions/workflows/ci.yml/runs?head_sha=base&status=success&per_page=5" in api.paths
    assert "/repos/o/r/actions/artifacts/50/zip" in api.paths


def test_fetch_baseline_degrades_to_a_marker_that_names_the_retention() -> None:
    api = _FakeApi(["base"], {}, {})
    marker = tor.fetch_baseline(api, "o/r", "base", "ci.yml", ["3.12"], 30)  # type: ignore[arg-type]
    assert marker["found"] is False
    assert "30 days" in marker["notice"]
    assert not tor.is_baseline_record(marker)


def test_fetch_baseline_degrades_when_the_artifact_is_not_a_zip() -> None:
    """A redirect body or a truncated download must not fail the step."""
    class _BadBytes(_FakeApi):
        def bytes(self, path: str) -> bytes:
            return b"<html>moved</html>"

    api = _BadBytes(["green"], {"green": [{"id": 5}]}, {5: [_artifact(50, "junit-3.12-m0")]})
    marker = tor.fetch_baseline(api, "o/r", "green", "ci.yml", ["3.12"], 30)  # type: ignore[arg-type]
    assert marker["found"] is False
    assert "artifact 50 of run 5 could not be read" in marker["notice"]


def test_junit_from_zip_requires_exactly_one_xml() -> None:
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a.xml", "<a />")
        zf.writestr("b.xml", "<b />")
    with pytest.raises(ValueError):
        tor.junit_from_zip(buf.getvalue())


# ------------------------------------------------------------------ CLI

def test_record_then_diff_through_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    junit = tmp_path / "junit-3.13.xml"
    junit.write_text(_MIXED, encoding="utf-8")
    current = tmp_path / "current.json"
    assert tor.main([
        "record", "--junit", f"3.13={junit}", "--junit", f"3.12={tmp_path / 'missing.xml'}",
        "--sha", "head", "--out", str(current),
    ]) == 0
    data = json.loads(current.read_text(encoding="utf-8"))
    assert data["legs"]["3.12"] is None and data["legs"]["3.13"]["counts"]["total"] == 10

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_record(**{"3.13": {"tests.test_a::test_skip": tor.PASSED}})))
    summary = tmp_path / "summary.md"
    capsys.readouterr()
    rc = tor.main(["diff", "--current", str(current), "--baseline", str(baseline), "--summary", str(summary)])
    assert rc == 0, "advisory mode never fails the step"
    out = capsys.readouterr().out
    assert "::warning::test outcome ratchet, leg 3.13: tests.test_a::test_skip passed -> skipped" in out
    text = summary.read_text(encoding="utf-8")
    assert "| `tests.test_a::test_skip` | passed -> skipped |" in text
    assert "leg 3.12 produced no JUnit report" in text

    rc = tor.main(["diff", "--current", str(current), "--baseline", str(baseline), "--block"])
    assert rc == 1, "blocking mode fails on a violation"


def test_block_mode_passes_without_violations_and_with_no_baseline(tmp_path: Path) -> None:
    current = tmp_path / "current.json"
    current.write_text(json.dumps(_record(**{"3.13": {"t::a": tor.PASSED}})))
    clean = tmp_path / "clean.json"
    clean.write_text(json.dumps(_record(**{"3.13": {"t::a": tor.PASSED}})))
    assert tor.main(["diff", "--current", str(current), "--baseline", str(clean), "--block"]) == 0
    marker = tmp_path / "marker.json"
    marker.write_text(json.dumps(tor.no_baseline_marker("expired")))
    summary = tmp_path / "summary.md"
    assert tor.main([
        "diff", "--current", str(current), "--baseline", str(marker), "--block", "--summary", str(summary),
    ]) == 0
    assert "No baseline: expired." in summary.read_text(encoding="utf-8")
    assert tor.main(["diff", "--current", str(current), "--baseline", str(tmp_path / "none"), "--block"]) == 0


def test_baseline_with_no_base_commit_writes_a_marker_without_the_api(tmp_path: Path) -> None:
    """A dispatch run has an empty `pull_request.base.sha`."""
    out = tmp_path / "baseline.json"
    assert tor.main(["baseline", "--repo", "o/r", "--base-sha", "", "--out", str(out)]) == 0
    marker = json.loads(out.read_text(encoding="utf-8"))
    assert marker["found"] is False
    assert "no base commit" in marker["notice"]


def test_an_unreadable_junit_is_a_usage_error(tmp_path: Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text("<testsuites><testsuite>", encoding="utf-8")
    assert tor.main(["record", "--junit", f"3.13={junit}", "--out", str(tmp_path / "o.json")]) == 2


# ------------------------------------------------------------- workflow

def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _job_lines(name: str) -> list[str]:
    lines = _CI.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"  {name}:")
    body: list[str] = []
    for line in lines[start + 1:]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if _indent(line) <= 2:
            break
        body.append(line)
    assert body, f"found `{name}:` under jobs but no body; the parser has rotted"
    return body


def _job_keys(name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in _job_lines(name):
        m = re.fullmatch(r"    ([A-Za-z-]+):\s*(.*?)\s*", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _steps(name: str) -> list[str]:
    """Each step of the job as one string."""
    steps: list[list[str]] = []
    for line in _job_lines(name):
        if _indent(line) == 6 and line.lstrip().startswith("- "):
            steps.append([line])
        elif steps:
            steps[-1].append(line)
    assert len(steps) >= 5, f"parsed only {len(steps)} steps; the parser has rotted"
    return ["\n".join(s) for s in steps]


def test_the_job_exists_runs_after_pytest_and_also_on_a_dispatch() -> None:
    """Guarding it to `pull_request` would make a dispatch emit a `skipped`
    row under this name, which clears an earlier failure (#1436 AC5)."""
    keys = _job_keys("test-outcomes")
    assert keys.get("needs") == "pytest"
    assert keys.get("if") == "${{ !cancelled() }}"
    assert keys.get("runs-on") == "ubuntu-latest"
    assert keys.get("timeout-minutes") == "10"


def test_the_job_is_advisory_by_the_2026_10_21_ruling() -> None:
    """Job-level `continue-on-error`, no `--block`, and a name outside the floor."""
    assert "    continue-on-error: true" in _job_lines("test-outcomes")
    assert "name" not in _job_keys("test-outcomes")
    assert "test-outcomes" not in FLOOR_NAMES
    diff_steps = [s for s in _steps("test-outcomes") if "test_outcome_record.py diff" in s]
    assert len(diff_steps) == 1
    assert "--block" not in diff_steps[0]
    assert "--allowlist .github/test-outcome-allowlist.txt" in diff_steps[0]
    assert '--summary "$GITHUB_STEP_SUMMARY"' in diff_steps[0]


def test_the_job_reads_other_runs_artifacts_with_a_read_token() -> None:
    body = "\n".join(_job_lines("test-outcomes"))
    assert "      actions: read" in body
    assert "      contents: read" in body
    baseline_steps = [s for s in _steps("test-outcomes") if "test_outcome_record.py baseline" in s]
    assert len(baseline_steps) == 1
    assert "GH_TOKEN: ${{ github.token }}" in baseline_steps[0]
    assert "BASE_SHA: ${{ github.event.pull_request.base.sha }}" in baseline_steps[0]
    assert '--base-sha "$BASE_SHA"' in baseline_steps[0]
    assert '[ -z "${BASE_SHA}" ]' in baseline_steps[0], "a dispatch has no base (#1436)"
    assert "write" not in body and "secrets." not in body
    # The 2026-10-21 ruling puts harden-runner only on jobs that hold a
    # write token or a secret; the two asserts above say this job holds none.
    assert "harden-runner" not in body


def test_the_job_downloads_this_runs_reports_by_the_merge_sha_pattern() -> None:
    steps = [s for s in _steps("test-outcomes") if "actions/download-artifact@" in s]
    assert len(steps) == 1
    assert f"actions/download-artifact@{_DOWNLOAD_SHA} # v4.3.0" in steps[0]
    assert "pattern: junit-*-${{ github.sha }}" in steps[0]
    assert "merge-multiple: true" in steps[0]
    record_steps = [s for s in _steps("test-outcomes") if "test_outcome_record.py record" in s]
    assert len(record_steps) == 1
    for leg in ("3.12", "3.13"):
        assert f"--junit {leg}=current/junit-{leg}.xml" in record_steps[0]
    assert "HEAD_SHA: ${{ github.event.pull_request.head.sha || github.sha }}" in record_steps[0]
    assert '--sha "$HEAD_SHA"' in record_steps[0]


def test_the_job_keeps_the_records_as_an_artifact() -> None:
    steps = [s for s in _steps("test-outcomes") if "actions/upload-artifact@" in s]
    assert len(steps) == 1
    assert f"actions/upload-artifact@{_UPLOAD_SHA} # v4.6.2" in steps[0]
    assert "name: test-outcome-record-${{ github.sha }}" in steps[0]
    assert "retention-days: 30" in steps[0]


def test_no_expression_is_interpolated_into_a_run_script() -> None:
    """zizmor template-injection: values reach shell through `env:`, never inline."""
    for step in _steps("test-outcomes"):
        run = re.search(r"\brun:(.*)", step, re.DOTALL)
        if run:
            assert "${{" not in run.group(1), step


def test_the_script_is_standard_library_only() -> None:
    """The job runs it on the runner's interpreter, without `uv sync`."""
    import ast
    tree = ast.parse((_REPO / "scripts" / "test_outcome_record.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert names <= set(sys.stdlib_module_names), names - set(sys.stdlib_module_names)
