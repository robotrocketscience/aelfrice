#!/usr/bin/env python3
"""Per-PR test outcome record and its diff against the last green main (#1786).

The `pytest` legs in `.github/workflows/ci.yml` upload a JUnit report each.
This script turns those reports into one record that names every test's
outcome, finds the record of the last green run on main, and diffs the two.
The `test-outcomes` job in `ci.yml` runs the three subcommands in order.

Usage:
    test_outcome_record.py record --junit LEG=PATH [--junit LEG=PATH ...]
        [--sha SHA] --out RECORD.json [--dry-run]
    test_outcome_record.py baseline --repo OWNER/NAME --base-sha SHA
        [--head-sha SHA] [--workflow ci.yml] [--legs 3.12,3.13] [--max-commits 30]
        --out RECORD.json [--dry-run]
    test_outcome_record.py diff --current RECORD.json --baseline RECORD.json
        --full-suite VALUE --pytest-result VALUE [--allowlist FILE] [--summary FILE]
        [--block] [--dry-run]

`--dry-run` writes no file and calls no API. `record` prints each leg's
counts, `baseline` prints the first request of its walk, and `diff` prints
the summary instead of appending it, with the exit status a real run gives.

Outcomes. Each test case in the report is one of `passed`, `failed`,
`error`, `skipped`, `xfailed`, `xpassed` or `never executed`. The last one
is a skip whose reason says the test is left out by design: the latency
tests behind `--run-perf` and the bench-gate tier that skips without the
lab corpus. A reason has to match one of those shapes whole, so an ordinary
skip that only mentions the flag or quotes a shape stays `skipped`, as does
a skip with any other reason. pytest writes an
xfail as `<skipped type="pytest.xfail">`, and a strict xpass as a failure
whose message starts with `[XPASS(strict)]`; `xfail_strict = true` in
pyproject.toml makes every xpass strict, so a lenient xpass, which pytest
records as a plain pass, does not occur in this suite.

The ratchet. A test whose outcome moves to `skipped`, `xfailed` or `never
executed` from any other outcome, or that is new as `skipped` or
`xfailed`, is a violation unless the allowlist names it. A new test that is
`never executed` is not one: the perf and bench-gate tests skip by design
on every CI leg (#1825). A skip reason matters only there, where one of the
exact shapes above makes a skip `never executed`. A platform skip is not one
of them, so a new `skipif(not WIN32)` test needs an allowlist entry (#1833).
A removed test is counted and is never a violation.

Renames. pytest renames parameter ids between releases, and pytest 9.2
renamed every id of a perf test (#1825). Within one test function, the id up
to its first `[`, the diff reads the new tests as renames of the removed
ones only when both groups hold the same outcomes the same number of times.
Any other mix pairs nothing: the new tests read as new and the removed ones
as removed, so a new skip or xfail in that function is a violation and an
allowlist entry is the way out. A new never-executed test is a violation
there too when the function lost a test with another outcome, since that
test may be the one that moved behind the gate. An even swap cannot be seen:
`f[a]` passing and `f[b]` skipped becoming `f[x]` skipped and `f[y]` passing
leaves the same outcomes, so it reads as a rename though a passing test may
have become a skip.

The allowlist file holds one test id per line, in which `*` matches any
run of characters and every other character, `[` and `?` included,
matches itself; blank lines and `#` comments are ignored.
Failures, errors and xpasses are reported but are not violations: the
pytest leg itself is red on those. The step summary names the first 50
tests of each list and stays under GitHub's 1 MiB limit, so a mass skip
still shows its counts; the record names every test. Without `--block` the exit status is 0
whatever the diff says and each violation is a `::warning::`; with it the
exit status is 1 when a violation exists and each is an `::error::`.
GitHub shows 10 annotations of a level per step, so one annotation that
counts the violations per leg comes before the per-test ones.

A baseline that holds some legs and not another compares the legs it holds.
The other leg gets a `::warning::` that names it and passes, since a leg can
be added on purpose.

A walk that finishes and finds no baseline is a `::warning::` and passes in
both modes: artifacts expire, and the first run has none. A lookup that
does not finish is a different case. On an API error, an answer without
the key it should hold, or an artifact or a report that cannot be read,
`baseline` writes a marker that says so and
exits 2, and `diff` exits 2 on that marker, on a baseline file that is
missing and on one that holds neither a record nor a no-baseline marker. A
docs-only run is the exception: it has no report to compare, so a failed
lookup there is a warning and passes.

A leg with no JUnit report is either a docs-only run or a lost report. Both
look the same in the record, and the `pytest` job's result is `success` in
both, so the job's `full_suite` output tells them apart. That output alone is
not enough: a leg that dies before the filter step computes `false`, and the
last leg to finish decides the value. So `diff` takes `--full-suite` and
`--pytest-result`, and a leg may lack a report only when they are exactly
`false` and `success`. In any other case `diff` exits 2 on a leg without one.

The baseline. `ci.yml` runs on `pull_request` and on `workflow_dispatch`,
never on a push to main. The merge train fast-forwards, so a merged PR's
head commit is the commit on main, and the run that tested it has that
commit as `head_sha`; a dispatched run has the commit it tested as
`head_sha` too. `baseline` walks main back from `--base-sha` and takes
the first commit with a successful run of the workflow, of either event,
that still holds a JUnit artifact for a leg. Artifacts are named
`junit-<leg>-<github.sha>`. On a `pull_request` run `github.sha` is the
merge commit of the PR, a commit that never lands on main, and on a
dispatch it is the tested commit, so artifacts are matched on the
`junit-<leg>-` prefix only.
Artifacts expire after 30 days and a docs-only PR uploads none; when the
walk finds no artifact the output says so and `diff` reports no baseline.
A dispatch carries an empty `--base-sha`, so `baseline` asks the compare API
for the merge base of main and `--head-sha` and walks from there. Without
that, a dispatch would report no baseline and pass, and the merge train,
which keeps the newest row per name, would read the pass over a pull
request's failure. With neither commit the lookup has failed.

Standard library only, so the job runs on the runner's own interpreter
without `uv sync`. The functions above `GhApi` are pure and take their
inputs as values; `tests/test_test_outcome_record_1786.py` runs them on
synthetic reports. `GhApi` calls `gh`, `fetch_baseline` calls it through
the client it is given, and the `cmd_*` functions read and write files
and print; the tests run those against a fake `gh` and temporary files.

Exit status: 0, or 1 for a ratchet violation under `--block`, or 2 for a
usage error, an unreadable input or a failed baseline lookup.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

RECORD_FORMAT = 1

PASSED = "passed"
FAILED = "failed"
ERROR = "error"
SKIPPED = "skipped"
XFAILED = "xfailed"
XPASSED = "xpassed"
NEVER_EXECUTED = "never executed"

OUTCOMES: tuple[str, ...] = (
    PASSED, FAILED, ERROR, SKIPPED, XFAILED, XPASSED, NEVER_EXECUTED,
)

# Moving a test to one of these, or adding a test with one of these, is
# what the ratchet refuses.
RATCHETED_OUTCOMES: frozenset[str] = frozenset({SKIPPED, XFAILED, NEVER_EXECUTED})

# Skip reasons that mean "left out by design", as the suite writes them.
# `--run-perf` is the opt-in registered in tests/conftest.py. The latency
# tests behind it write their own skip reason, in these wordings; a test
# pins this set against every perf-gated skip in tests/.
PERF_SKIP_REASONS: frozenset[str] = frozenset({
    "perf benchmark; pass --run-perf to enable",
    "perf test gated on --run-perf",
    "perf-gated: pass --run-perf to run",
})
# The bench-gate tier's two shapes, copied because this script runs without
# the suite's dependencies: `BENCH_GATE_SKIP_REASON` and the reason
# `skip_corpus_module` writes for each of `CORPUS_SKIP_STATES`, in
# tests/conftest.py. A test pins these to that file. The corpus root is an
# absolute path on every leg that can reach the shape.
BENCH_GATE_SKIP_REASON = (
    "AELFRICE_CORPUS_ROOT not set or not a directory; skipping bench-gate test "
    "(lab corpus absent by design \u2014 see #1420 \u00a73; the corpus is private and "
    "this repository is public)"
)
CORPUS_SKIP_STATES: tuple[str, ...] = ("empty", "missing", "underfilled", "ungradeable")
_CORPUS_SKIP_REASON = re.compile(rf"corpus module '[^']+' (?:{'|'.join(CORPUS_SKIP_STATES)}) under /.*")

_XPASS_STRICT_PREFIX = "[XPASS(strict)]"
_XPASS_LEGACY_MESSAGE = "xfail-marked test passes unexpectedly"

_API_TIMEOUT_SECONDS = 60

# GitHub refuses a step summary over 1 MiB and then shows none of it, the
# counts table included, so the summary stays under that with room to spare.
SUMMARY_MAX_BYTES = 1_000_000
# Tests each list of a leg names before it says how many it left out. A
# mass skip moves every test in the suite, thousands of rows per leg.
SUMMARY_MAX_ROWS = 50
_ROWS_LEFT_OUT = "{n} more not listed; this run's `test-outcome-record` artifact names every test's outcome."
_SUMMARY_CUT = (
    "The summary stops here to stay under {limit} bytes; "
    "this run's `test-outcome-record` artifact names every test's outcome."
)


# ---------------------------------------------------------------- record

def classify_testcase(testcase: ElementTree.Element) -> str:
    """One outcome for the children of a test's `<testcase>` entries.

    A failure outranks an error, so a test that failed and then errored in
    teardown reads as failed. An error outranks a skip, so a test that
    skipped and then errored in teardown reads as an error.
    """
    failure = testcase.find("failure")
    if failure is not None:
        if (failure.get("message") or "").startswith(_XPASS_STRICT_PREFIX):
            return XPASSED
        return FAILED
    if testcase.find("error") is not None:
        return ERROR
    skipped = testcase.find("skipped")
    if skipped is None:
        return PASSED
    message = skipped.get("message") or ""
    if skipped.get("type") == "pytest.xfail":
        return XFAILED
    if message == _XPASS_LEGACY_MESSAGE:
        return XPASSED
    if is_never_executed_reason(message):
        return NEVER_EXECUTED
    return SKIPPED


def is_never_executed_reason(reason: str) -> bool:
    return (
        reason in PERF_SKIP_REASONS
        or reason == BENCH_GATE_SKIP_REASON
        or _CORPUS_SKIP_REASON.fullmatch(reason) is not None
    )


def test_id(testcase: ElementTree.Element) -> str:
    """`classname::name`, the way the terminal summary names a test."""
    return f"{testcase.get('classname') or ''}::{testcase.get('name') or ''}"


def parse_junit(text: str) -> dict[str, str]:
    """Test id to outcome for one JUnit report.

    pytest writes a test whose call failed and whose teardown then raised
    as two `<testcase>` entries with one id, the first holding `<failure>`
    and the second `<error>`. The children of every entry with one id are
    classified together, so the precedence in `classify_testcase` holds
    across entries.
    """
    root = ElementTree.fromstring(text)
    merged: dict[str, ElementTree.Element] = {}
    for testcase in root.iter("testcase"):
        merged.setdefault(test_id(testcase), ElementTree.Element("testcase")).extend(list(testcase))
    return {tid: classify_testcase(entry) for tid, entry in merged.items()}


def count_outcomes(outcomes: dict[str, str]) -> dict[str, int]:
    counts = {o: 0 for o in OUTCOMES}
    for outcome in outcomes.values():
        counts[outcome] += 1
    counts["total"] = len(outcomes)
    return counts


def build_record(legs: dict[str, str | None], sha: str | None) -> dict[str, Any]:
    """The record for one run: leg name to the JUnit text, or None for a leg
    that produced no report (the suite did not run on that leg)."""
    record: dict[str, Any] = {"format": RECORD_FORMAT, "sha": sha, "legs": {}}
    for leg, text in sorted(legs.items()):
        if text is None:
            record["legs"][leg] = None
            continue
        outcomes = parse_junit(text)
        record["legs"][leg] = {
            "tests": dict(sorted(outcomes.items())),
            "counts": count_outcomes(outcomes),
        }
    return record


# ------------------------------------------------------------------ diff

@dataclass(frozen=True)
class Transition:
    test: str
    before: str | None
    after: str | None


@dataclass
class LegDiff:
    leg: str
    counts: dict[str, int]
    baseline_present: bool
    violations: list[Transition] = field(default_factory=list)
    recovered: list[Transition] = field(default_factory=list)
    red: list[Transition] = field(default_factory=list)
    added: int = 0
    removed: int = 0
    renamed: int = 0
    notice: str | None = None
    # The run has a baseline and it holds no record of this leg.
    baseline_lacks_leg: bool = False


def parse_allowlist(text: str) -> list[str]:
    patterns: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(line)
    return patterns


def allowlist_regex(entry: str) -> re.Pattern[str]:
    """An allowlist entry as a regex: `*` matches any run of characters and
    every other character matches itself. Test ids hold `[`, `?` and `|` in
    their parameters, so no other glob syntax applies."""
    return re.compile(".*".join(re.escape(part) for part in entry.split("*")))


def is_allowlisted(test: str, allowlist: Iterable[str]) -> bool:
    return any(allowlist_regex(entry).fullmatch(test) for entry in allowlist)


def function_id(test: str) -> str:
    """The test function an id belongs to: the id up to its first `[`, so
    every parameter of one function shares it."""
    return test.partition("[")[0]


def _by_function(outcomes: dict[str, str]) -> dict[str, list[str]]:
    """Each test function's ids, ordered by outcome and then by id."""
    groups: dict[str, list[str]] = {}
    for test in sorted(outcomes, key=lambda t: (outcomes[t], t)):
        groups.setdefault(function_id(test), []).append(test)
    return groups


def pair_renames(added: dict[str, str], removed: dict[str, str]) -> dict[str, str]:
    """Each new test id the diff reads as a rename, to the removed id it renames.

    Both arguments map a test id to its outcome. A function's ids pair only
    when its new and removed tests hold the same outcomes the same number of
    times, so a pair always has one outcome on both sides.
    """
    old_groups = _by_function(removed)
    pairs: dict[str, str] = {}
    for function, new in _by_function(added).items():
        old = old_groups.get(function, [])
        if [added[t] for t in new] == [removed[t] for t in old]:
            pairs.update(zip(new, old))
    return pairs


def is_violation(before: str | None, after: str | None) -> bool:
    """The ratchet rule on one transition. `before` is None for a new test
    the diff did not pair with a removed one."""
    if before is None and after == NEVER_EXECUTED:
        return False
    return after in RATCHETED_OUTCOMES and before != after


def diff_leg(
    leg: str,
    current: dict[str, str],
    baseline: dict[str, str] | None,
    allowlist: Iterable[str] = (),
) -> LegDiff:
    allow = list(allowlist)
    out = LegDiff(leg=leg, counts=count_outcomes(current), baseline_present=baseline is not None)
    for test in sorted(current):
        after = current[test]
        if after in (FAILED, ERROR, XPASSED):
            out.red.append(Transition(test, None if baseline is None else baseline.get(test), after))
    if baseline is None:
        out.notice = f"no baseline for leg {leg}; the ratchet was not evaluated"
        return out
    removed = {test: outcome for test, outcome in baseline.items() if test not in current}
    renames = pair_renames({t: o for t, o in current.items() if t not in baseline}, removed)
    out.renamed = len(renames)
    # Functions that lost a test which was not already left out by design: a
    # new never-executed test there that did not pair may be that test behind
    # the gate.
    lost = {function_id(test) for test, outcome in removed.items() if outcome != NEVER_EXECUTED}
    for test in sorted(set(current) | set(baseline)):
        before = baseline.get(test)
        after = current.get(test)
        if before is None:
            out.added += 1
        if after is None:
            out.removed += 1
            continue
        if before == after or test in renames:
            continue
        t = Transition(test, before, after)
        moved_behind_gate = before is None and after == NEVER_EXECUTED and function_id(test) in lost
        if (is_violation(before, after) or moved_behind_gate) and not is_allowlisted(test, allow):
            out.violations.append(t)
        elif before in RATCHETED_OUTCOMES and after == PASSED:
            out.recovered.append(t)
    return out


def diff_records(
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    allowlist: Iterable[str] = (),
) -> list[LegDiff]:
    """One LegDiff per leg of the current record, in leg order."""
    allow = list(allowlist)
    diffs: list[LegDiff] = []
    for leg, leg_record in sorted(current.get("legs", {}).items()):
        if leg_record is None:
            d = LegDiff(leg=leg, counts=count_outcomes({}), baseline_present=False)
            d.notice = f"leg {leg} produced no JUnit report"
            diffs.append(d)
            continue
        base_leg = None if baseline is None else baseline.get("legs", {}).get(leg)
        base_tests = None if base_leg is None else base_leg.get("tests", {})
        d = diff_leg(leg, leg_record["tests"], base_tests, allow)
        d.baseline_lacks_leg = baseline is not None and base_leg is None
        diffs.append(d)
    return diffs


def has_violations(diffs: Iterable[LegDiff]) -> bool:
    return any(d.violations for d in diffs)


def missing_report_error(current: dict[str, Any]) -> str | None:
    """What is wrong with a record from a run that was meant to report on
    every leg, or None when nothing is."""
    legs: dict[str, Any] = current.get("legs") or {}
    if not legs:
        return "this run's record holds no leg"
    missing = sorted(leg for leg, record in legs.items() if record is None)
    if not missing:
        return None
    return f"the full suite ran and no JUnit report reached this job for leg {', '.join(missing)}"


def _describe(t: Transition) -> str:
    return f"{t.before or 'absent'} -> {t.after or 'absent'}"


def _table_cell(text: str) -> str:
    """`text` with each `|` escaped. A GFM table splits a row on every
    unescaped `|`, inside a code span too, and some test ids hold one."""
    return text.replace("|", "\\|")


def _capped(rows: list[str]) -> list[str]:
    """The first `SUMMARY_MAX_ROWS` of `rows`, then a paragraph naming how
    many it left out."""
    if len(rows) <= SUMMARY_MAX_ROWS:
        return [*rows, ""]
    left_out = len(rows) - SUMMARY_MAX_ROWS
    return [*rows[:SUMMARY_MAX_ROWS], "", _ROWS_LEFT_OUT.format(n=left_out), ""]


def _fit(text: str) -> str:
    """`text` within `SUMMARY_MAX_BYTES` of UTF-8: when it is longer, its
    whole lines up to that size and a closing note."""
    if len(text.encode("utf-8")) <= SUMMARY_MAX_BYTES:
        return text
    note = "\n" + _SUMMARY_CUT.format(limit=SUMMARY_MAX_BYTES) + "\n"
    room = SUMMARY_MAX_BYTES - len(note.encode("utf-8"))
    head = text.encode("utf-8")[:room].decode("utf-8", "ignore")
    return head[: head.rfind("\n") + 1] + note


def render_summary(
    diffs: list[LegDiff],
    baseline: dict[str, Any] | None,
    baseline_notice: str | None,
    block: bool,
    lookup_error: str | None = None,
    report_error: str | None = None,
) -> str:
    """The step-summary Markdown, at most `SUMMARY_MAX_BYTES` long. Each
    list of a leg names its first `SUMMARY_MAX_ROWS` tests; the counts
    table and each list's total always show."""
    lines = [f"## Test outcomes ({'blocking' if block else 'advisory'})", ""]
    if report_error is not None:
        lines.extend([f"No report to compare: {report_error}. This check fails.", ""])
    if lookup_error is not None:
        lines.append(f"No usable baseline: {lookup_error}. The ratchet was not evaluated, so this check fails.")
    elif baseline is None:
        lines.append(f"No baseline: {baseline_notice or 'none given'}. The ratchet was not evaluated.")
    else:
        lines.append(f"Baseline: main @ `{baseline.get('sha') or 'unknown'}`.")
    lines.append("")
    header = ["Leg", *OUTCOMES, "total"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for d in diffs:
        cells = [d.leg, *(str(d.counts[o]) for o in OUTCOMES), str(d.counts["total"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    for d in diffs:
        lines.append(f"### Leg {d.leg}")
        lines.append("")
        if d.notice:
            lines.append(d.notice + ".")
            lines.append("")
        if d.red:
            lines.append(f"{len(d.red)} red outcome(s) in this run (the pytest leg reports these):")
            lines.append("")
            lines.extend(_capped([f"- `{t.test}`: {t.after}" for t in d.red]))
        if not d.baseline_present:
            continue
        lines.append(
            f"{len(d.violations)} ratchet violation(s), {len(d.recovered)} recovered, "
            f"{d.added} added, {d.removed} removed, {d.renamed} renamed."
        )
        lines.append("")
        if d.violations:
            lines.append("| Test | Outcome |")
            lines.append("|---|---|")
            lines.extend(_capped([
                f"| `{_table_cell(t.test)}` | {_describe(t)} |" for t in d.violations
            ]))
        if d.recovered:
            lines.append("Recovered:")
            lines.append("")
            lines.extend(_capped([f"- `{t.test}`: {_describe(t)}" for t in d.recovered]))
    return _fit("\n".join(lines).rstrip() + "\n")


def render_annotations(
    diffs: list[LegDiff],
    baseline_notice: str | None,
    block: bool,
    allowlist: str | None = None,
    lookup_error: str | None = None,
    report_error: str | None = None,
) -> list[str]:
    """Workflow commands: an error for a lost report and one for a failed
    lookup, one that counts the violations, then one per violation, and one
    for a missing baseline. GitHub shows 10 annotations of a level per step,
    so the ones that explain the failure go first and a long list cannot
    hide them. `allowlist` is the path the count names."""
    level = "error" if block else "warning"
    out: list[str] = []
    if report_error is not None:
        out.append(f"::error::test outcomes: {report_error}")
    if lookup_error is not None:
        # A command ends at its line; gh's stderr can span several.
        out.append(f"::error::test outcomes: no usable baseline: {' '.join(lookup_error.split())}")
    per_leg = [f"leg {d.leg}: {len(d.violations)}" for d in diffs if d.violations]
    if per_leg:
        total = sum(len(d.violations) for d in diffs)
        entry = f"an entry in {allowlist}" if allowlist else "an allowlist entry"
        out.append(
            f"::{level}::test outcome ratchet: {total} violation(s) ({', '.join(per_leg)}). "
            f"The step summary names the first {SUMMARY_MAX_ROWS} of each leg and the record artifact names "
            f"every test. A test that is meant to skip needs {entry}."
        )
    if baseline_notice:
        out.append(f"::warning::test outcomes: no baseline: {baseline_notice}")
    for d in diffs:
        if d.baseline_lacks_leg:
            out.append(f"::warning::test outcomes: no baseline for leg {d.leg}; the ratchet was not evaluated on it")
    for d in diffs:
        for t in d.violations:
            out.append(f"::{level}::test outcome ratchet, leg {d.leg}: {t.test} {_describe(t)}")
    return out


# -------------------------------------------------------------- baseline

@dataclass(frozen=True)
class BaselineRef:
    sha: str
    run_id: int
    artifacts: dict[str, int]


def artifact_leg(name: str, legs: Iterable[str]) -> str | None:
    """The leg an artifact name `junit-<leg>-<sha>` belongs to, or None."""
    for leg in legs:
        if name.startswith(f"junit-{leg}-"):
            return leg
    return None


def choose_baseline(
    commits: list[str],
    runs_for: Callable[[str], list[dict[str, Any]]],
    artifacts_for: Callable[[int], list[dict[str, Any]]],
    legs: Iterable[str],
) -> BaselineRef | None:
    """The first commit, in order, with a successful run holding a live JUnit
    artifact for at least one leg. `runs_for` returns the successful runs
    whose `head_sha` is the commit, newest first."""
    leg_list = list(legs)
    for sha in commits:
        for run in runs_for(sha):
            found: dict[str, int] = {}
            for artifact in artifacts_for(int(run["id"])):
                if artifact.get("expired"):
                    continue
                leg = artifact_leg(str(artifact.get("name", "")), leg_list)
                if leg is not None and leg not in found:
                    found[leg] = int(artifact["id"])
            if found:
                return BaselineRef(sha=sha, run_id=int(run["id"]), artifacts=found)
    return None


def junit_from_zip(data: bytes) -> str:
    """The one `.xml` member of an artifact zip."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".xml")]
        if len(names) != 1:
            raise ValueError(f"expected one .xml in the artifact, found {names}")
        return zf.read(names[0]).decode("utf-8")


def no_baseline_marker(notice: str) -> dict[str, Any]:
    """The walk finished and found no record to compare with."""
    return {"format": RECORD_FORMAT, "found": False, "notice": notice}


def lookup_error_marker(notice: str) -> dict[str, Any]:
    """The lookup did not finish, so a baseline may exist that it never read."""
    return {**no_baseline_marker(notice), "error": True}


def is_baseline_record(data: dict[str, Any]) -> bool:
    return data.get("found", True) is not False and "legs" in data


def is_lookup_error(data: dict[str, Any]) -> bool:
    return data.get("error") is True


class GhApi:
    """`gh api` as a function of the path, with a timeout on every call."""

    def json(self, path: str) -> Any:
        return json.loads(self._run(path))

    def bytes(self, path: str) -> bytes:
        return self._run(path)

    @staticmethod
    def _run(path: str) -> bytes:
        r = subprocess.run(
            ["gh", "api", path],
            capture_output=True,
            check=False,
            timeout=_API_TIMEOUT_SECONDS,
        )
        if r.returncode != 0:
            raise RuntimeError(f"gh api {path} failed: {r.stderr.decode('utf-8', 'replace').strip()}")
        return r.stdout


def commits_path(repo: str, base_sha: str, max_commits: int) -> str:
    """The first request of the walk: main's history below the base commit."""
    return f"/repos/{repo}/commits?sha={base_sha}&per_page={max_commits}"


def merge_base_path(repo: str, head_sha: str) -> str:
    """The compare request whose answer names the merge base. `per_page=1`
    keeps the response small; the merge base does not depend on it."""
    return f"/repos/{repo}/compare/main...{head_sha}?per_page=1"


def fetch_merge_base(api: GhApi, repo: str, head_sha: str) -> str:
    return str(api.json(merge_base_path(repo, head_sha))["merge_base_commit"]["sha"])


def fetch_baseline(
    api: GhApi,
    repo: str,
    base_sha: str,
    workflow: str,
    legs: list[str],
    max_commits: int,
) -> dict[str, Any]:
    """The baseline record, a no-baseline marker when the walk finds none,
    or a lookup-error marker when an artifact or a report cannot be read."""
    commits_json = api.json(commits_path(repo, base_sha, max_commits))
    commits = [c["sha"] for c in commits_json]

    def runs_for(sha: str) -> list[dict[str, Any]]:
        data = api.json(
            f"/repos/{repo}/actions/workflows/{workflow}/runs"
            f"?head_sha={sha}&status=success&per_page=5"
        )
        return list(data["workflow_runs"])

    def artifacts_for(run_id: int) -> list[dict[str, Any]]:
        data = api.json(f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100")
        return list(data["artifacts"])

    ref = choose_baseline(commits, runs_for, artifacts_for, legs)
    if ref is None:
        return no_baseline_marker(
            f"no successful {workflow} run with a JUnit artifact on the last "
            f"{len(commits)} commits of main below {base_sha[:12]} "
            "(artifacts expire after 30 days)"
        )
    texts: dict[str, str | None] = {leg: None for leg in legs}
    for leg, artifact_id in ref.artifacts.items():
        data = api.bytes(f"/repos/{repo}/actions/artifacts/{artifact_id}/zip")
        try:
            texts[leg] = junit_from_zip(data)
        except (zipfile.BadZipFile, ValueError) as exc:
            return lookup_error_marker(
                f"artifact {artifact_id} of run {ref.run_id} could not be read: {exc}"
            )
    try:
        record = build_record(texts, ref.sha)
    except ElementTree.ParseError as exc:
        return lookup_error_marker(f"a JUnit report of run {ref.run_id} could not be parsed: {exc}")
    record["run_id"] = ref.run_id
    return record


# ------------------------------------------------------------------ main

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _parse_leg_arg(value: str) -> tuple[str, Path]:
    leg, sep, path = value.partition("=")
    if not sep or not leg or not path:
        raise argparse.ArgumentTypeError(f"expected LEG=PATH, got {value!r}")
    return leg, Path(path)


def cmd_record(args: argparse.Namespace) -> int:
    legs: dict[str, str | None] = {}
    for leg, path in args.junit:
        if path.is_file():
            legs[leg] = path.read_text(encoding="utf-8")
        else:
            print(f"::notice::no JUnit report for leg {leg} at {path}", file=sys.stderr)
            legs[leg] = None
    record = build_record(legs, args.sha)
    if args.dry_run:
        print(f"dry run: would write {args.out}")
        for leg, data in record["legs"].items():
            if data is None:
                print(f"leg {leg}: no JUnit report")
                continue
            counts = data["counts"]
            print(f"leg {leg}: " + ", ".join(f"{o} {counts[o]}" for o in (*OUTCOMES, "total")))
        return 0
    _write_json(args.out, record)
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    legs = [leg for leg in args.legs.split(",") if leg]
    if args.dry_run:
        if args.base_sha:
            print(f"dry run: would walk main with gh api {commits_path(args.repo, args.base_sha, args.max_commits)}")
        elif args.head_sha:
            print(f"dry run: would find the base commit with gh api {merge_base_path(args.repo, args.head_sha)}")
        else:
            print("dry run: no base commit and no head commit, so no API call")
        print(f"dry run: would write {args.out}")
        return 0
    if not args.base_sha and not args.head_sha:
        record = lookup_error_marker("this run has no base commit and no head commit to find one from")
    else:
        try:
            api = GhApi()
            # A `workflow_dispatch` run carries no pull request, so its base
            # is the merge base of main and the commit it tested.
            base_sha = args.base_sha or fetch_merge_base(api, args.repo, args.head_sha)
            record = fetch_baseline(api, args.repo, base_sha, args.workflow, legs, args.max_commits)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, AttributeError, subprocess.TimeoutExpired) as exc:
            record = lookup_error_marker(f"the baseline lookup failed: {exc}")
    _write_json(args.out, record)
    # A log line only: the diff step reads the marker and writes the annotation.
    if record.get("found") is False:
        print(record["notice"], file=sys.stderr)
    return 2 if is_lookup_error(record) else 0


def cmd_diff(args: argparse.Namespace) -> int:
    current = _read_json(args.current)
    reports_optional = args.full_suite == "false" and args.pytest_result == "success"
    report_error = None if reports_optional else missing_report_error(current)
    baseline: dict[str, Any] | None = None
    baseline_notice: str | None = None
    lookup_error: str | None = None
    if args.baseline is None:
        baseline_notice = "no baseline file given"
    elif not args.baseline.is_file():
        lookup_error = f"baseline file {args.baseline} is missing"
    else:
        data = _read_json(args.baseline)
        if is_baseline_record(data):
            baseline = data
        elif data.get("found") is False and not is_lookup_error(data):
            baseline_notice = str(data.get("notice") or "no baseline record")
        else:
            lookup_error = str(data.get("notice") or f"baseline file {args.baseline} holds no record")
    if lookup_error is not None and reports_optional and not any((current.get("legs") or {}).values()):
        # A docs-only run: no report, so the failed lookup cost no comparison.
        baseline_notice, lookup_error = lookup_error, None
    allowlist = parse_allowlist(args.allowlist.read_text(encoding="utf-8")) if args.allowlist else []
    diffs = diff_records(current, baseline, allowlist)
    summary = render_summary(diffs, baseline, baseline_notice, args.block, lookup_error, report_error)
    if args.summary is not None and not args.dry_run:
        with args.summary.open("a", encoding="utf-8") as fh:
            fh.write(summary)
    else:
        if args.summary is not None:
            print(f"dry run: would append the summary to {args.summary}")
        sys.stdout.write(summary)
    allowlist_name = str(args.allowlist) if args.allowlist else None
    for line in render_annotations(diffs, baseline_notice, args.block, allowlist_name, lookup_error, report_error):
        print(line)
    if lookup_error is not None or report_error is not None:
        return 2
    return 1 if args.block and has_violations(diffs) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="write no file, call no API, print what would happen")

    p = sub.add_parser("record", parents=[common], help="build a record from JUnit reports")
    p.add_argument("--junit", action="append", required=True, type=_parse_leg_arg, metavar="LEG=PATH")
    p.add_argument("--sha", default=None)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("baseline", parents=[common], help="fetch the last green main record through gh api")
    p.add_argument("--repo", required=True)
    p.add_argument("--base-sha", required=True)
    p.add_argument("--head-sha", default="")
    p.add_argument("--workflow", default="ci.yml")
    p.add_argument("--legs", default="3.12,3.13")
    p.add_argument("--max-commits", type=int, default=30)
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("diff", parents=[common], help="diff a record against a baseline record")
    p.add_argument("--current", required=True, type=Path)
    p.add_argument("--baseline", default=None, type=Path)
    p.add_argument(
        "--full-suite", required=True,
        help="the pytest job's full_suite output",
    )
    p.add_argument(
        "--pytest-result", required=True,
        help="the pytest job's result; a leg may lack a report only with `--full-suite false` and `success` here",
    )
    p.add_argument("--allowlist", default=None, type=Path)
    p.add_argument("--summary", default=None, type=Path)
    p.add_argument("--block", action="store_true")
    p.set_defaults(func=cmd_diff)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, KeyError, ElementTree.ParseError) as exc:
        # An annotation, so a check that fails here says why on the run's page.
        print(f"::error::test_outcome_record: {' '.join(str(exc).split())}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
