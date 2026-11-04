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
import subprocess
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

# Sha pins and the releases they name. The release is not pinned:
# Dependabot bumps both, and tests/test_workflow_supply_chain_1786.py holds
# the pin rule for every `uses:` line in the tree.
_DOWNLOAD_PIN = re.compile(r"actions/download-artifact@[0-9a-f]{40} # v\d+\.\d+\.\d+")
_UPLOAD_PIN = re.compile(r"actions/upload-artifact@[0-9a-f]{40} # v\d+\.\d+\.\d+")
_USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(.*?)\s*$", re.MULTILINE)


def _uses(step: str) -> str:
    """The one `uses:` value in a step."""
    values = _USES.findall(step)
    assert len(values) == 1, step
    return values[0]


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


_PERF_SKIP_WORDINGS = (
    "perf test gated on --run-perf",
    "perf benchmark; pass --run-perf to enable",
    "perf-gated: pass --run-perf to run",
)


@pytest.mark.parametrize("reason", _PERF_SKIP_WORDINGS)
def test_every_perf_skip_wording_in_the_suite_is_never_executed(reason: str) -> None:
    """The latency tests word the reason in more than one way; all name the flag."""
    assert tor.is_never_executed_reason(reason)


# The two shapes the suite gates a latency test on: the `_has_run_perf`
# helper, and a `run_perf` local read from `--run-perf`.
_PERF_GATE_NAMES = frozenset({"_has_run_perf", "run_perf"})


def _skip_reason(node: Any) -> str | None:
    """The reason of a `pytest.skip(...)` call, `""` for a non-literal one,
    and None for any other node."""
    import ast
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
        return None
    if node.func.attr != "skip" or not node.args:
        return None
    arg = node.args[0]
    return arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else ""


def _perf_skip_sites_in_the_suite() -> tuple[dict[str, str], dict[str, str]]:
    """Every `pytest.skip(...)` under tests/ as `path:line -> reason`, twice:
    the sites inside an `if` on a perf gate, and the sites whose reason
    names `--run-perf`."""
    import ast
    gated: dict[str, str] = {}
    naming: dict[str, str] = {}
    for path in sorted((_REPO / "tests").rglob("*.py")):
        if path.name == Path(__file__).name:
            continue
        rel = path.relative_to(_REPO).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            reason = _skip_reason(node)
            if reason is not None and "--run-perf" in reason:
                naming[f"{rel}:{node.lineno}"] = reason
            if not isinstance(node, ast.If):
                continue
            names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if not names & _PERF_GATE_NAMES:
                continue
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    reason = _skip_reason(sub)
                    if reason is not None:
                        gated[f"{rel}:{sub.lineno}"] = reason
    return gated, naming


def test_the_perf_wordings_above_are_every_one_the_suite_uses() -> None:
    """A new wording, or a site that stops naming the flag, shows up here.

    The sites are found by their gate, not by their wording, so a gated
    skip whose reason drops `--run-perf` fails the second assert even when
    other sites still use its old wording."""
    gated, naming = _perf_skip_sites_in_the_suite()
    assert len(gated) >= len(_PERF_SKIP_WORDINGS), gated
    assert {site: r for site, r in gated.items() if not tor.is_never_executed_reason(r)} == {}
    assert gated == naming
    assert set(gated.values()) == set(_PERF_SKIP_WORDINGS)


@pytest.mark.parametrize("reason", [
    "needs sqlite 3.45",
    "requires the 'archive' extra",
    "perf",
    "bench-gate",
])
def test_an_ordinary_skip_reason_is_not_never_executed(reason: str) -> None:
    assert not tor.is_never_executed_reason(reason)


def _bench_gate_skip_reasons() -> dict[str, str]:
    """The bench-gate skip reasons as tests/conftest.py writes them: the
    whole-tier `BENCH_GATE_SKIP_REASON`, and one `skip_corpus_module`
    reason per corpus state it accepts, keyed by a short label."""
    from tests.conftest import BENCH_GATE_SKIP_REASON, CORPUS_SKIP_STATES, skip_corpus_module

    reasons = {"whole-tier": BENCH_GATE_SKIP_REASON}
    for why in sorted(CORPUS_SKIP_STATES):
        with pytest.raises(pytest.skip.Exception) as info:
            skip_corpus_module("retrieve", why, Path("/corpus"), "a detail")
        reasons[f"module-{why}"] = str(info.value.msg)
    return reasons


_BENCH_GATE_REASONS = _bench_gate_skip_reasons()


@pytest.mark.parametrize("reason", list(_BENCH_GATE_REASONS.values()), ids=list(_BENCH_GATE_REASONS))
def test_every_bench_gate_skip_conftest_writes_is_never_executed(reason: str) -> None:
    """Built from the conftest text, not a copy of it: a reworded reason
    would otherwise turn every bench-gate skip into `skipped`."""
    from xml.sax.saxutils import escape, quoteattr

    body = f'<skipped type="pytest.skip" message={quoteattr(reason)}>t.py:3: {escape(reason)}</skipped>'
    outcomes = tor.parse_junit(_report(_case("tests.bench_gate.test_b", "test_gate", body)))
    assert outcomes == {"tests.bench_gate.test_b::test_gate": tor.NEVER_EXECUTED}


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


_LITERAL_IDS = [
    "tests.test_store.TestOpen::test_readonly[a]",
    "tests.t::test_x[p-1]",
    "tests.t::test_x[!a]",
    "tests.t::test_x[/tmp/a b]",
    "tests.t::test_x[Owner | 3 +]",
    "tests.t::test_x[||||||| main]",
    "tests.t::test_x[a?b]",
    "tests.t::test_x[a.*b]",
    "tests.t::test_x[^$\\d+]",
]


@pytest.mark.parametrize("test", _LITERAL_IDS)
def test_an_allowlist_entry_that_is_a_test_id_matches_that_id(test: str) -> None:
    """The allowlist header tells the operator to write the id as the record names it."""
    assert tor.is_allowlisted(test, [test])
    d = tor.diff_leg("3.12", {test: tor.SKIPPED}, {test: tor.PASSED}, [test])
    assert d.violations == []


def test_only_a_star_in_an_allowlist_entry_is_a_wildcard() -> None:
    assert not tor.is_allowlisted("tests.test_store.TestOpen::test_readonlya", ["tests.test_store.TestOpen::test_readonly[a]"])
    assert not tor.is_allowlisted("tests.t::test_x[ab]", ["tests.t::test_x[a?]"])
    assert not tor.is_allowlisted("tests.t::test_x[a]", ["tests.t::test_x"])
    assert not tor.is_allowlisted("tests.t::test_x", ["tests.t::test_x[a]"])
    assert tor.is_allowlisted("tests.t::test_x[p-1]", ["tests.t::test_x[*]"])
    assert tor.is_allowlisted("tests.t::test_x[p-12]", ["tests.t::test_x[*]"])
    assert tor.is_allowlisted("tests.t::test_x[]", ["tests.t::test_x[*]"])
    assert not tor.is_allowlisted("tests.t::test_y[p-1]", ["tests.t::test_x[*]"])


def _is_one_test_id(entry: str) -> bool:
    """`::` present, no whitespace in the classname or the function name,
    and no comment after the id: `parse_allowlist` keeps a same-line `# ...`
    as part of the entry, which then matches nothing. A parameter may hold
    whitespace and `#` (`[Let's move on to PR #869]`), so the check looks for
    the comment after the closing bracket, not anywhere in the entry."""
    classname, sep, name = entry.partition("::")
    function = name.partition("[")[0]
    return (
        bool(sep)
        and not re.search(r"\s", classname + function)
        and not re.search(r"\]\s+#", entry)
    )


@pytest.mark.parametrize(
    ("entry", "ok"),
    [
        ("tests.test_store.TestOpen::test_readonly[a]", True),
        ("tests.test_ingest_pasted_output_1691::test_prose_that_opens_like_output_is_kept[Owner | 3 +]", True),
        ("tests.t::test_x[/tmp/a b]", True),
        ("tests.test_win*::*", True),
        ("tests.test_store", False),
        ("tests.test_store test_readonly", False),
        ("tests.test_store # why::test_readonly", False),
        ("tests.test_store.TestOpen::test_readonly[a]  # skips on 3.13", False),
        ("tests.test_store.TestOpen::test_readonly[a]\t# skips on 3.13", False),
        ("tests.t::test_x[a#b]", True),
        ("tests.test_cadence_p2::test_phase_boundary_prefix_match[Let's move on to PR #869]", True),
        (
            "tests.bench_gate.test_commit_intent::test_classifies_corrective_messages"
            "[Fix #1234: handle empty input-fix]",
            True,
        ),
        ("tests.t::test_x  # skips on 3.13", False),
        ("tests.test_win*::*  # skips on the linux legs", False),
        ("tests.t::test_x[a]  # see [b]", False),
    ],
)
def test_the_entry_shape_check(entry: str, ok: bool) -> None:
    """Parameters may hold spaces and `#`; the classname and the function
    name never hold whitespace, and `::` must be there."""
    assert _is_one_test_id(entry) is ok


def test_every_checked_in_allowlist_entry_is_one_test_id() -> None:
    """A bare module name, or an id with a comment after it, would parse and
    match nothing. The checked-in file may hold no entry yet, so the next
    test runs this check on a file that does."""
    for entry in tor.parse_allowlist(_ALLOWLIST.read_text(encoding="utf-8")):
        assert _is_one_test_id(entry), entry


def test_the_shape_check_refuses_the_entry_the_allowlist_header_warns_about() -> None:
    """The header says a comment after the id on the same line becomes part
    of the id. Parsed as the ratchet parses it, that entry exempts nothing,
    and the shape check above refuses it."""
    test = "tests.test_store.TestOpen::test_readonly[a]"
    text = _ALLOWLIST.read_text(encoding="utf-8") + f"# why it skips\n{test}\n{test}  # skips on 3.13\n"
    entries = tor.parse_allowlist(text)
    assert entries == [test, f"{test}  # skips on 3.13"]
    assert [_is_one_test_id(e) for e in entries] == [True, False]
    assert not tor.is_allowlisted(test, entries[1:])


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


def test_diff_records_passes_the_allowlist_to_every_leg() -> None:
    current = _record(**{"3.12": {"t::a": tor.SKIPPED}, "3.13": {"t::a": tor.SKIPPED}})
    baseline = _record(**{"3.12": {"t::a": tor.PASSED}, "3.13": {"t::a": tor.PASSED}})
    assert tor.has_violations(tor.diff_records(current, baseline))
    assert not tor.has_violations(tor.diff_records(current, baseline, ["t::*"]))


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


@pytest.mark.parametrize("with_baseline", [True, False], ids=["baseline", "no-baseline"])
def test_the_summary_lists_each_red_outcome_with_or_without_a_baseline(with_baseline: bool) -> None:
    """Failures, errors and xpasses are reported, though they are not violations."""
    tests = {"t::f": tor.FAILED, "t::e": tor.ERROR, "t::x": tor.XPASSED, "t::p": tor.PASSED, "t::s": tor.SKIPPED}
    current = _record(**{"3.13": tests})
    baseline = _record(**{"3.13": tests}) if with_baseline else None
    text = tor.render_summary(tor.diff_records(current, baseline), baseline, None, block=False)
    section = text.split("### Leg 3.13", 1)[1]
    assert "3 red outcome(s) in this run (the pytest leg reports these):" in section
    bullets = [line for line in section.splitlines() if line.startswith("- ")]
    assert bullets == ["- `t::e`: error", "- `t::f`: failed", "- `t::x`: xpassed"]


def _unescaped_pipes(line: str) -> int:
    return len(re.findall(r"(?<!\\)\|", line))


def test_a_pipe_in_a_test_id_keeps_the_violation_row_at_two_cells() -> None:
    """GFM splits a table cell on `|` even inside a code span unless it is `\\|`."""
    test = "tests.test_conflict_markers_1491::test_diff3_style_base_marker_is_caught[||||||| main]"
    current = _record(**{"3.13": {test: tor.SKIPPED}})
    baseline = _record(**{"3.13": {test: tor.PASSED}})
    text = tor.render_summary(tor.diff_records(current, baseline), baseline, None, block=False)
    escaped = test.replace("|", "\\|")
    assert f"| `{escaped}` | passed -> skipped |" in text
    rows = [line for line in text.splitlines() if line.startswith("|")]
    table = rows[rows.index("| Test | Outcome |"):]
    assert len(table) == 3 and all(_unescaped_pipes(row) == 3 for row in table)


_MASS_IDS = [
    f"tests.test_module_{i // 100:03d}.TestClassWithALongName::test_case_with_a_long_name_{i}[param-{i}]"
    for i in range(13_000)
]


@pytest.mark.parametrize("after", [tor.SKIPPED, tor.ERROR], ids=["mass-skip", "mass-error"])
def test_a_mass_change_keeps_the_summary_under_githubs_limit(after: str) -> None:
    """GitHub drops a step summary over 1 MiB whole, counts table included."""
    passed = dict.fromkeys(_MASS_IDS, tor.PASSED)
    moved = dict.fromkeys(_MASS_IDS, after)
    baseline = _record(**{"3.12": passed, "3.13": passed})
    current = _record(**{"3.12": moved, "3.13": moved})
    text = tor.render_summary(tor.diff_records(current, baseline), baseline, None, block=False)
    assert len(text.encode("utf-8")) <= tor.SUMMARY_MAX_BYTES < 1024 * 1024
    for leg in ("3.12", "3.13"):
        assert f"| {leg} |" in text
        section = text.split(f"### Leg {leg}", 1)[1].split("### Leg", 1)[0]
        rows = [line for line in section.splitlines() if line.startswith(("- `", "| `"))]
        assert len(rows) == tor.SUMMARY_MAX_ROWS
        assert f"{len(_MASS_IDS) - tor.SUMMARY_MAX_ROWS} more not listed;" in section
    if after == tor.SKIPPED:
        assert "13000 ratchet violation(s)" in text
    else:
        assert "13000 red outcome(s)" in text


def test_a_short_list_names_every_test_and_no_left_out_line() -> None:
    ids = _MASS_IDS[: tor.SUMMARY_MAX_ROWS]
    current = _record(**{"3.13": dict.fromkeys(ids, tor.SKIPPED)})
    baseline = _record(**{"3.13": dict.fromkeys(ids, tor.PASSED)})
    text = tor.render_summary(tor.diff_records(current, baseline), baseline, None, block=False)
    assert sum(line.startswith("| `") for line in text.splitlines()) == tor.SUMMARY_MAX_ROWS
    assert "more not listed" not in text


def test_a_summary_past_the_byte_cap_ends_on_a_whole_line_and_a_note() -> None:
    """Long ids on many legs pass the byte cap even with each list capped."""
    ids = [f"tests.t::test_x[{i}-{'p' * 4_000}]" for i in range(tor.SUMMARY_MAX_ROWS)]
    legs = {f"3.{n}": dict.fromkeys(ids, tor.SKIPPED) for n in range(10)}
    base = {leg: dict.fromkeys(ids, tor.PASSED) for leg in legs}
    current, baseline = _record(**legs), _record(**base)
    text = tor.render_summary(tor.diff_records(current, baseline), baseline, None, block=False)
    assert len(text.encode("utf-8")) <= tor.SUMMARY_MAX_BYTES
    assert text.startswith("## Test outcomes (advisory)\n")
    assert "| 3.0 |" in text and "| 3.9 |" in text
    lines = text.splitlines()
    assert lines[-1].startswith("The summary stops here to stay under 1000000 bytes;")
    assert lines[-2] == "" and lines[-3].endswith("| passed -> skipped |")


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


def _zip(*names: str) -> bytes:
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in names:
            zf.writestr(name, "<a />")
    return buf.getvalue()


@pytest.mark.parametrize("names", [(), ("notes.txt",), ("a.xml", "b.xml")], ids=["empty", "no-xml", "two-xml"])
def test_junit_from_zip_requires_exactly_one_xml(names: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        tor.junit_from_zip(_zip(*names))
    assert tor.junit_from_zip(_zip("notes.txt", "junit.xml")) == "<a />"


def test_fetch_baseline_degrades_when_the_artifact_holds_no_report() -> None:
    class _NoXml(_FakeApi):
        def bytes(self, path: str) -> bytes:
            return _zip("notes.txt")

    api = _NoXml(["green"], {"green": [{"id": 5}]}, {5: [_artifact(50, "junit-3.12-m0")]})
    marker = tor.fetch_baseline(api, "o/r", "green", "ci.yml", ["3.12"], 30)  # type: ignore[arg-type]
    assert marker["found"] is False
    assert "artifact 50 of run 5 could not be read" in marker["notice"]


class _BadXml(_FakeApi):
    """An artifact that unzips fine and holds one report that does not parse."""

    def __init__(self, xml: str) -> None:
        super().__init__(["green"], {"green": [{"id": 5}]}, {5: [_artifact(50, "junit-3.12-m0")]})
        self.xml = xml

    def bytes(self, path: str) -> bytes:
        import io
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("junit-3.12.xml", self.xml)
        return buf.getvalue()


_UNPARSEABLE = ["<testsuites><testsuite><testcase", "not xml"]


@pytest.mark.parametrize("xml", _UNPARSEABLE, ids=["truncated", "not-xml"])
def test_fetch_baseline_degrades_when_the_report_does_not_parse(xml: str) -> None:
    marker = tor.fetch_baseline(_BadXml(xml), "o/r", "green", "ci.yml", ["3.12"], 30)  # type: ignore[arg-type]
    assert marker["found"] is False
    assert "a JUnit report of run 5 could not be parsed" in marker["notice"]


@pytest.mark.parametrize("xml", _UNPARSEABLE, ids=["truncated", "not-xml"])
def test_an_unparseable_baseline_report_writes_a_marker_and_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], xml: str,
) -> None:
    """The diff step after it then reports "no baseline" rather than never running."""
    monkeypatch.setattr(tor, "GhApi", lambda: _BadXml(xml))
    out = tmp_path / "baseline.json"
    argv = ["baseline", "--repo", "o/r", "--base-sha", "green", "--legs", "3.12", "--out", str(out)]
    assert tor.main(argv) == 0
    marker = json.loads(out.read_text(encoding="utf-8"))
    assert marker["found"] is False
    assert "::notice::a JUnit report of run 5 could not be parsed" in capsys.readouterr().err


def test_fetch_baseline_asks_for_max_commits_of_history() -> None:
    api = _FakeApi(["base"], {}, {})
    tor.fetch_baseline(api, "o/r", "base", "ci.yml", ["3.12"], 7)  # type: ignore[arg-type]
    assert api.paths[0] == "/repos/o/r/commits?sha=base&per_page=7"


class _FakeGh:
    """Stands in for `subprocess.run` under GhApi and records each call."""

    def __init__(self, returncode: int, stdout: bytes, stderr: bytes = b"") -> None:
        self.result = (returncode, stdout, stderr)
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((cmd, kwargs))
        returncode, stdout, stderr = self.result
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def test_gh_api_runs_gh_api_on_the_path_with_a_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    gh = _FakeGh(0, b'{"workflow_runs": [{"id": 7}]}', b"a warning")
    monkeypatch.setattr(tor.subprocess, "run", gh)
    assert tor.GhApi().json("/repos/o/r/x") == {"workflow_runs": [{"id": 7}]}
    assert tor.GhApi().bytes("/repos/o/r/y") == b'{"workflow_runs": [{"id": 7}]}'
    kwargs = {"capture_output": True, "check": False, "timeout": tor._API_TIMEOUT_SECONDS}
    assert gh.calls == [(["gh", "api", "/repos/o/r/x"], kwargs), (["gh", "api", "/repos/o/r/y"], kwargs)]
    assert 0 < tor._API_TIMEOUT_SECONDS <= 120


def test_gh_api_raises_with_gh_stderr_when_gh_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tor.subprocess, "run", _FakeGh(1, b"{}", b"HTTP 404: Not Found\n"))
    with pytest.raises(RuntimeError, match=r"^gh api /repos/o/r/x failed: HTTP 404: Not Found$"):
        tor.GhApi().json("/repos/o/r/x")
    with pytest.raises(RuntimeError):
        tor.GhApi().bytes("/repos/o/r/x")


def test_a_gh_failure_through_gh_api_becomes_a_no_baseline_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tor.subprocess, "run", _FakeGh(1, b"", b"HTTP 502"))
    out = tmp_path / "baseline.json"
    assert tor.main(["baseline", "--repo", "o/r", "--base-sha", "base", "--out", str(out)]) == 0
    marker = json.loads(out.read_text(encoding="utf-8"))
    assert marker["found"] is False and "HTTP 502" in marker["notice"]


# ------------------------------------------------------------------ CLI

def test_record_then_diff_through_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    junit = tmp_path / "junit-3.13.xml"
    junit.write_text(_MIXED, encoding="utf-8")
    current = tmp_path / "current.json"
    missing = tmp_path / "missing.xml"
    capsys.readouterr()
    assert tor.main([
        "record", "--junit", f"3.13={junit}", "--junit", f"3.12={missing}",
        "--sha", "head", "--out", str(current),
    ]) == 0
    assert capsys.readouterr().err.splitlines() == [f"::notice::no JUnit report for leg 3.12 at {missing}"]
    data = json.loads(current.read_text(encoding="utf-8"))
    assert data["legs"]["3.12"] is None and data["legs"]["3.13"]["counts"]["total"] == 10

    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_record(**{"3.13": {"tests.test_a::test_skip": tor.PASSED}})))
    summary = tmp_path / "summary.md"
    rc = tor.main(["diff", "--current", str(current), "--baseline", str(baseline), "--summary", str(summary)])
    assert rc == 0, "advisory mode never fails the step"
    out = capsys.readouterr().out
    assert "::warning::test outcome ratchet, leg 3.13: tests.test_a::test_skip passed -> skipped" in out
    text = summary.read_text(encoding="utf-8")
    assert "| `tests.test_a::test_skip` | passed -> skipped |" in text
    assert "leg 3.12 produced no JUnit report" in text

    rc = tor.main(["diff", "--current", str(current), "--baseline", str(baseline), "--block"])
    assert rc == 1, "blocking mode fails on a violation"


def test_the_allowlist_flag_reaches_the_ratchet(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The operator's one exemption path, from `--allowlist` to each leg's diff."""
    current = tmp_path / "current.json"
    current.write_text(json.dumps(_record(**{
        "3.12": {"tests.test_win_x::t": tor.SKIPPED, "tests.test_posix::t": tor.PASSED},
        "3.13": {"tests.test_win_x::t": tor.SKIPPED, "tests.test_posix::t": tor.PASSED},
    })))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_record(**{
        "3.12": {"tests.test_win_x::t": tor.PASSED, "tests.test_posix::t": tor.PASSED},
        "3.13": {"tests.test_win_x::t": tor.PASSED, "tests.test_posix::t": tor.PASSED},
    })))
    allowlist = tmp_path / "allow.txt"
    allowlist.write_text("# windows-only\ntests.test_win*::*\n", encoding="utf-8")
    base_args = ["diff", "--current", str(current), "--baseline", str(baseline), "--block"]

    assert tor.main(base_args) == 1
    capsys.readouterr()
    assert tor.main([*base_args, "--allowlist", str(allowlist)]) == 0
    out = capsys.readouterr().out
    assert "::warning::" not in out and "::error::" not in out
    assert out.count("0 ratchet violation(s)") == 2


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


def test_baseline_passes_its_flags_to_the_walk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    api = _FakeApi(
        ["base"],
        {"base": [{"id": 7}]},
        {7: [_artifact(70, "junit-3.12-m1"), _artifact(71, "junit-3.13-m1")]},
    )
    monkeypatch.setattr(tor, "GhApi", lambda: api)
    out = tmp_path / "baseline.json"
    assert tor.main([
        "baseline", "--repo", "o/r", "--base-sha", "base", "--workflow", "w.yml",
        "--legs", "3.13", "--max-commits", "7", "--out", str(out),
    ]) == 0
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["sha"] == "base" and list(record["legs"]) == ["3.13"]
    assert api.paths[0] == "/repos/o/r/commits?sha=base&per_page=7"
    assert "/repos/o/r/actions/workflows/w.yml/runs?head_sha=base&status=success&per_page=5" in api.paths


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("gh api failed: HTTP 502"),
        ValueError("bad value"),
        KeyError("sha"),
        json.JSONDecodeError("Expecting value", "", 0),
        subprocess.TimeoutExpired(["gh", "api"], 60),
    ],
    ids=["gh-failed", "value", "key", "json", "timeout"],
)
def test_a_failed_baseline_lookup_writes_a_marker_and_exits_0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], exc: Exception,
) -> None:
    class _Failing(_FakeApi):
        def json(self, path: str) -> Any:
            raise exc

    monkeypatch.setattr(tor, "GhApi", lambda: _Failing([], {}, {}))
    out = tmp_path / "baseline.json"
    assert tor.main(["baseline", "--repo", "o/r", "--base-sha", "base", "--out", str(out)]) == 0
    marker = json.loads(out.read_text(encoding="utf-8"))
    assert marker["found"] is False
    assert marker["notice"].startswith("the baseline lookup failed: ")
    assert "::notice::the baseline lookup failed" in capsys.readouterr().err


def test_dry_run_record_prints_the_counts_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    junit = tmp_path / "junit-3.13.xml"
    junit.write_text(_MIXED, encoding="utf-8")
    out = tmp_path / "current.json"
    assert tor.main([
        "record", "--junit", f"3.13={junit}", "--junit", f"3.12={tmp_path / 'missing.xml'}",
        "--out", str(out), "--dry-run",
    ]) == 0
    assert not out.exists()
    text = capsys.readouterr().out
    assert f"dry run: would write {out}" in text
    assert "leg 3.12: no JUnit report" in text
    assert "leg 3.13: passed 3, failed 1, error 1, skipped 1, xfailed 1, xpassed 1, never executed 2, total 10" in text


@pytest.mark.parametrize("base_sha", ["base", ""], ids=["pull-request", "dispatch"])
def test_dry_run_baseline_calls_no_api_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], base_sha: str,
) -> None:
    def _no_api() -> None:
        raise AssertionError("a dry run built an API client")

    monkeypatch.setattr(tor, "GhApi", _no_api)
    out = tmp_path / "baseline.json"
    assert tor.main([
        "baseline", "--repo", "o/r", "--base-sha", base_sha, "--max-commits", "7", "--out", str(out), "--dry-run",
    ]) == 0
    assert not out.exists()
    text = capsys.readouterr().out
    assert f"dry run: would write {out}" in text
    if base_sha:
        assert "gh api /repos/o/r/commits?sha=base&per_page=7" in text
    else:
        assert "no base commit" in text


def test_dry_run_diff_prints_the_summary_appends_nothing_and_keeps_the_exit_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    current = tmp_path / "current.json"
    current.write_text(json.dumps(_record(**{"3.13": {"t::a": tor.SKIPPED}})))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(_record(**{"3.13": {"t::a": tor.PASSED}})))
    summary = tmp_path / "summary.md"
    summary.write_text("earlier step\n", encoding="utf-8")
    rc = tor.main([
        "diff", "--current", str(current), "--baseline", str(baseline),
        "--summary", str(summary), "--block", "--dry-run",
    ])
    assert rc == 1
    assert summary.read_text(encoding="utf-8") == "earlier step\n"
    text = capsys.readouterr().out
    assert f"dry run: would append the summary to {summary}" in text
    assert "## Test outcomes (blocking)" in text
    assert "::error::test outcome ratchet, leg 3.13: t::a passed -> skipped" in text


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
    assert keys.get("timeout-minutes") == "15"


def test_every_step_times_out_before_the_job_does() -> None:
    """A job that reaches its own `timeout-minutes` ends `cancelled`, and so
    does the workflow run, job-level `continue-on-error` or not. The
    baseline walk reads only successful runs, so it would skip that commit.
    A step past its own cap ends `failure`, which the step's
    `continue-on-error` turns into `success`. So each step carries a cap,
    and the caps add up to less than the job's."""
    caps: list[int] = []
    for step in _steps("test-outcomes"):
        m = re.search(r"\n        timeout-minutes: (\d+)\n", step + "\n")
        assert m, step
        caps.append(int(m.group(1)))
    assert sum(caps) < int(_job_keys("test-outcomes")["timeout-minutes"]), caps


def test_the_job_is_advisory_by_the_2026_10_21_ruling() -> None:
    """`continue-on-error` on every step, no `--block`, and a name outside the floor.

    The job-level line keeps the workflow run green, but the job's check-run
    still reads `failure` when a step fails, and `merge_train_gate.evaluate`
    counts that check-run as failing. A step-level line is what keeps the
    check-run at `success`, so every step carries one."""
    for step in _steps("test-outcomes"):
        assert "\n        continue-on-error: true" in step, step
    assert _job_keys("test-outcomes").get("continue-on-error") == "true"
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
    assert _DOWNLOAD_PIN.fullmatch(_uses(steps[0])), steps[0]
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
    assert _UPLOAD_PIN.fullmatch(_uses(steps[0])), steps[0]
    assert "name: test-outcome-record-${{ github.sha }}" in steps[0]
    assert "retention-days: 30" in steps[0]


def _one_step(marker: str) -> str:
    steps = [s for s in _steps("test-outcomes") if marker in s]
    assert len(steps) == 1, marker
    return steps[0]


def _flag(step: str, flag: str) -> str:
    values = re.findall(rf"{re.escape(flag)} (\S+)", step)
    assert len(values) == 1, (flag, step)
    return values[0]


def _block_lines(step: str, key: str) -> list[str]:
    """The lines of a `key: |` block scalar in a step."""
    lines = step.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == f"{key}: |")
    out: list[str] = []
    for line in lines[start + 1:]:
        if _indent(line) <= _indent(lines[start]):
            break
        out.append(line.strip())
    return out


def test_each_step_reads_the_files_the_step_before_it_wrote() -> None:
    """The job is advisory, so a broken handoff would stay green in CI."""
    download = _one_step("actions/download-artifact@")
    record = _one_step("test_outcome_record.py record")
    baseline = _one_step("test_outcome_record.py baseline")
    diff = _one_step("test_outcome_record.py diff")
    upload = _one_step("actions/upload-artifact@")
    path = re.search(r"^ +path: (\S+)$", download, re.MULTILINE)
    assert path, "without `path` the reports land in the workspace root"
    for leg in ("3.12", "3.13"):
        assert f"--junit {leg}={path.group(1)}/junit-{leg}.xml" in record
    assert _flag(diff, "--current") == _flag(record, "--out")
    assert _flag(diff, "--baseline") == _flag(baseline, "--out")
    assert sorted(_block_lines(upload, "path")) == sorted([_flag(record, "--out"), _flag(baseline, "--out")])
    assert _flag(baseline, "--repo") == '"$GITHUB_REPOSITORY"'


def _ci_triggers() -> list[str]:
    """The event names under ci.yml's top-level `on:`."""
    lines = _CI.read_text(encoding="utf-8").splitlines()
    triggers: list[str] = []
    for line in lines[lines.index("on:") + 1:]:
        if line and not line.startswith(" "):
            break
        m = re.match(r"  ([a-z_]+):", line)
        if m:
            triggers.append(m.group(1))
    return triggers


def _ci_baseline_comment() -> str:
    """The ci.yml comment paragraph that opens with `The baseline.`."""
    lines = _CI.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("  # The baseline."))
    block: list[str] = []
    for line in lines[start:]:
        if not line.startswith("  #") or not line[3:].strip():
            break
        block.append(line[3:].strip())
    return " ".join(block)


def test_the_baseline_notes_name_every_event_the_workflow_runs_on() -> None:
    """The walk reads green runs of any event; a note that names one event
    misstates which `head_sha` and `github.sha` a run carries."""
    triggers = _ci_triggers()
    assert "pull_request" in triggers and len(triggers) > 1, triggers
    assert tor.__doc__ is not None
    docstring = next(p for p in tor.__doc__.split("\n\n") if p.startswith("The baseline."))
    for text in (docstring, _ci_baseline_comment()):
        for event in triggers:
            assert f"`{event}`" in text, (event, text)


def test_no_expression_is_interpolated_into_a_run_script() -> None:
    """zizmor template-injection: values reach shell through `env:`, never inline."""
    for step in _steps("test-outcomes"):
        run = re.search(r"\brun:(.*)", step, re.DOTALL)
        if run:
            assert "${{" not in run.group(1), step


_IO_NAMES = frozenset({"print", "open", "input", "subprocess", "sys", "os", "Path"})
_IO_ATTRS = frozenset({"read_text", "write_text", "read_bytes", "write_bytes", "open", "run", "write", "unlink"})


def test_the_functions_the_docstring_calls_pure_do_no_io() -> None:
    """The module docstring names the first definition below the pure ones;
    nothing above it may print, open a file or start a process."""
    import ast
    assert tor.__doc__ is not None
    m = re.search(r"The functions above `(\w+)` are pure", tor.__doc__)
    assert m, "the docstring no longer says which functions are pure"
    tree = ast.parse((_REPO / "scripts" / "test_outcome_record.py").read_text(encoding="utf-8"))
    defs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    names = [n.name for n in defs]
    assert m.group(1) in names, m.group(1)
    impure: list[str] = []
    for node in defs[: names.index(m.group(1))]:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in _IO_NAMES:
                impure.append(f"{node.name}: {sub.id}")
            elif isinstance(sub, ast.Attribute) and sub.attr in _IO_ATTRS:
                impure.append(f"{node.name}: .{sub.attr}")
    assert impure == []
    assert len(defs[: names.index(m.group(1))]) > 10


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
