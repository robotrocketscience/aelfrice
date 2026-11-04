#!/usr/bin/env python3
"""The nightly canary's record, triage input and results branch (#1786).

The nightly canary runs the full suite on main every night, on two canary
legs (Python 3.12 and 3.13) and on extra legs that are tracked but never
open an issue. Each leg reruns its failures once in the same job, so a
failure that passes on the rerun is flaky and one that fails again is
confirmed. This script turns each leg's bundle into one record, finds the
confirmed failures that are new since the last green night, and writes the
night to the `test-results` branch. It does not run pytest, git bisect or
`gh`; the workflow and `scripts/nightly_bisect.py` do.

Usage:
    nightly_canary.py env --out ENV.json [--dry-run]
    nightly_canary.py failed-ids --junit JUNIT.xml --repo-root DIR --out IDS.txt
        [--max 50] [--github-output FILE] [--dry-run]
    nightly_canary.py verdict --rc RC --junit JUNIT.xml
        [--rerun RERUN.xml --rerun-rc RC] [--max 50] [--out VERDICT.json] [--dry-run]
    nightly_canary.py record --leg LEG=DIR[:ROLE] [--leg ...] --sha SHA
        --run-id ID --run-attempt N [--event NAME] [--created-at ISO8601]
        [--canary-legs 3.12,3.13] --out RECORD.json [--dry-run]
    nightly_canary.py regressions --current RECORD.json --last-green-dir DIR
        [--canary-legs 3.12,3.13] --out TRIAGE.json [--dry-run]
    nightly_canary.py publish --record RECORD.json --triage TRIAGE.json
        --branch-dir DIR [--main-repo .] [--main-ref REF] [--dry-run]

`--dry-run` writes no file and changes no git state. Each subcommand then
prints what it would write, with the exit status a real run gives.
`publish` still runs `git rev-parse` and `git merge-base --is-ancestor`,
which read only.

A leg's bundle. The suite job of a leg writes `env.json` (from `env`, run
under the leg's interpreter with `uv run`), `junit.xml` from the full run,
`failed-ids.txt` and `rerun.xml` when the full run exited 1, and
`verdict.json` (from `verdict --out`). It uploads them as the artifact
`nightly-<leg>-<sha>-a<run_attempt>`.

`failed-ids` names the tests to rerun. A JUnit test id is
`classname::name`, with the module path dotted in `classname`. The node id
takes the one prefix of `classname` that is a file `<repo-root>/a/b.py`; the
rest of `classname` is class names, and `name` keeps its parameter brackets.
An id maps to no node id when it would not round-trip through pytest: its
module no longer exists, it is a collection error (pytest writes those with
an empty `classname`), two prefixes are files (a class sharing its name with
a file in a same-stem directory), the path holds `[` or `]` (pytest refuses
those), or a directory holds a dot (its parts read as class names). One such
id in the file would make pytest exit 4 for the whole rerun, so each is left
out and named in a `::notice::`, and the verdict counts it as confirmed. The
exit status carries the rerun decision:
0 means the file holds at least one node id, and 1 means there is nothing to
rerun (no mapped red test, or more than `--max` red tests) and the file does
not exist, a stale one included. No empty file is ever written, because
pytest given an empty `@file` runs the whole suite and given a missing one
exits 4. The decision also prints as `rerun=true|false`, and goes to
`--github-output` when that is given. The rerun step runs only on exit 0.

`verdict` reads one leg. Red outcomes are `failed`, `error` and `xpassed`,
the outcomes that turn a pytest leg red. A red test whose rerun outcome is
`passed` is flaky; every other red test is confirmed. More than `--max` red
tests is a mass failure: the rerun is skipped and every red test is
confirmed. A rerun counts only when pytest exited 0 or 1 and left a readable
report. `--rerun` and `--rerun-rc` go together, since exit 4 (an id pytest
cannot resolve) can leave a report that says nothing about the others. When
the rerun does not count, every red test stays confirmed, `rerun_note` in
the verdict says why, and the verdict is still written. The leg's status is `green` when nothing is confirmed and pytest
exited 0 or 1, `red` when a test is confirmed, and `broken` when nothing is
confirmed and pytest exited with any other status, or with 1 and no red
test, or the report holds no test. The exit status is 1 unless the leg is
green. XML that is not a JUnit report, such as an HTML error page, is an
unreadable input.

`record` writes a format-1 record that `test_outcome_record.py` reads,
with every leg, plus a `nightly` key holding the run, each leg's role,
environment, rc, status, flaky and confirmed tests, notes saying why a leg
reads as it does, and the night's verdict. `DIR` is either one leg's bundle or the directory the artifacts
were downloaded into; in the second case the bundle is the subdirectory
`nightly-<leg>-<sha>-a<N>` with the highest `N`. `ROLE` is `canary`
(the default) or `extra`. One bad bundle never loses the night's record: a
leg is `missing`, with a note saying why, when `junit.xml` or `verdict.json`
is absent or unreadable, when `env.json` is unreadable, or when
`verdict.json` disagrees with `junit.xml` (a green verdict over a report
with a failure, for one). A leg in `--canary-legs` that no `--leg` names
reads as missing too. The night is `green` when every canary leg is green and
every leg in `--canary-legs` is a canary leg, `incomplete` when one of those
is missing or there is no canary leg at all, and `red` otherwise. `--created-at` defaults to the current
UTC time and `--event` to `GITHUB_EVENT_NAME`.

`regressions` compares the canary legs of the record with the last green
record that `last-green.json` in `--last-green-dir` points to. A leg named
in `--canary-legs` that the record gives the `extra` role is not a canary
leg, and neither is a leg the flag leaves out: extra legs never produce an
event. The output carries:

- `events`: `canary-broken` when a canary leg is missing or broken;
  `no-baseline-<sha12>` when there is no last green night yet;
  `drift-<lastgreen12>` when tonight's commit is the last green commit,
  since only the environment can have changed. Each of the last two holds
  every confirmed test.
- `candidates`: confirmed tests that the last green record holds on the
  same leg, with their outcome there. A test that failed there and was
  flaky counts as `passed`.
- `new_failing`: confirmed tests the last green record does not hold.
- `drift_check`: the candidates the workflow reruns at the last green
  commit, with the Python of the failing leg (the newest one when both
  fail); a mass failure samples 20 of them.

`publish` writes the night to a checkout of the results branch:
`runs/<YYYY-MM-DD>-<run_id>-a<run_attempt>/record.json` and `triage.json`,
one line in `index.jsonl` per night, and `last-green.json`. Each attempt
of a run gets its own directory, so a red re-attempt never overwrites the
green record `last-green.json` points at. The index keeps one line per
run, the newest attempt's, and names that attempt's directory. A missing
triage file is a notice: the night's record is still published, with no
event keys.

The `canary` entry of `last-green.json` moves only on a green night and each
`legs` entry only when that leg is green. Either moves only to a commit on
main that descends from the old one, the first pointer included. Main is
`--main-ref` in `--main-repo`, by default `origin/main` or else `main`. A
green night on a commit that is not on main, or that git cannot place, moves
nothing, and `publish` exits 1. A later commit already named stays named,
which is the ordinary case of a late run: that exits 0. At one commit, the
newer run and attempt hold the pointer, so a late older attempt cannot take
it. An entry holds `attempt` beside `run_id` for that comparison. The record
an entry names does not change: republishing the same run and attempt with
different content exits 1 and writes nothing. The same content is a no-op.

A pointer can fall off main through a history rewrite or a hand edit. Such a
pointer would freeze, since no commit descends from it on main, so `publish`
exits 1 on every night that leaves it in place, naming it. The next green
night on main re-points it with a `::warning::` and exits 0. An entry whose
record file is gone exits 1 the same way, naming the path, until a green
night moves it. Exit 1 comes after the writes, so the workflow can commit the
night's record before it fails the job.

Standard library only, so the publish job runs it on the runner's own
interpreter. It imports `parse_junit`, `build_record` and `diff_records`
from `test_outcome_record.py` next to it. The functions above `uv_version`
are pure and take their inputs as values; `tests/test_nightly_canary_1786.py`
runs them on synthetic reports and records, and runs the `cmd_*` functions
against temporary files and a temporary git repository.

Exit status: 0; 1 for a checked condition (`failed-ids` has nothing to
rerun, `verdict` finds the leg is not green, `publish` left a pointer problem
behind); 2 for a usage error or an unreadable input.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent))

# A dry run writes no file, and that includes `__pycache__` beside this script.
_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
import test_outcome_record as tor  # noqa: E402

sys.dont_write_bytecode = _write_bytecode

TRIAGE_FORMAT = 1

#: Outcomes that turn a pytest leg red (`LegDiff.red` in test_outcome_record).
RED_OUTCOMES: frozenset[str] = frozenset({tor.FAILED, tor.ERROR, tor.XPASSED})

#: More red tests than this on one leg is a mass failure: no rerun.
MASS_FAILURE_THRESHOLD = 50

#: Candidates the drift check runs after a mass failure.
MASS_FAILURE_SAMPLE = 20

#: The legs a night needs before it can read as green.
DEFAULT_CANARY_LEGS: tuple[str, ...] = ("3.12", "3.13")

CANARY = "canary"
EXTRA = "extra"
ROLES: tuple[str, ...] = (CANARY, EXTRA)

GREEN = "green"
RED = "red"
BROKEN = "broken"
MISSING = "missing"
INCOMPLETE = "incomplete"

#: pytest exit statuses for "all passed" and "some tests failed".
_TEST_RC = (0, 1)

_BUNDLE_FILES = ("junit.xml", "verdict.json", "env.json")

_GIT_TIMEOUT_SECONDS = 60
_UV_TIMEOUT_SECONDS = 30

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


# --------------------------------------------------------------- failed-ids

def parse_report(text: str) -> dict[str, str]:
    """`parse_junit`, refusing XML whose root is not a JUnit one: it holds
    no `<testcase>`, so it would read as a run with nothing wrong."""
    root = ElementTree.fromstring(text)
    if root.tag not in ("testsuites", "testsuite"):
        raise ValueError(f"not a JUnit report: the root element is <{root.tag}>")
    return tor.parse_junit(text)


def red_ids(outcomes: dict[str, str]) -> list[str]:
    """The test ids with a red outcome, sorted."""
    return sorted(t for t, o in outcomes.items() if o in RED_OUTCOMES)


def node_id(test: str, is_file: Callable[[str], bool]) -> str | None:
    """The pytest node id for a JUnit `classname::name`, or None.

    `is_file` says whether a path relative to the repo root is a file. The
    module is the one dotted prefix of `classname` that names a `.py` file;
    the parts after it are classes. None when pytest could not select the
    result: no such prefix, more than one, a bracket in the path, or a part
    after the module that cannot be a class name.
    """
    classname, sep, name = test.partition("::")
    if not sep or not classname or not name or "[" in classname or "]" in classname:
        return None
    parts = classname.split(".")
    cuts = [cut for cut in range(1, len(parts) + 1) if is_file("/".join(parts[:cut]) + ".py")]
    if len(cuts) != 1 or not all(part.isidentifier() for part in parts[cuts[0]:]):
        return None
    return "::".join(["/".join(parts[:cuts[0]]) + ".py", *parts[cuts[0]:], name])


def plan_rerun(
    outcomes: dict[str, str], is_file: Callable[[str], bool], max_ids: int,
) -> dict[str, Any]:
    """What the rerun step runs: node ids, the ids left unmapped, and
    whether to rerun at all."""
    red = red_ids(outcomes)
    if len(red) > max_ids:
        return {"count": len(red), "rerun": False, "mass_failure": True, "ids": [], "unmapped": []}
    ids: list[str] = []
    unmapped: list[str] = []
    for test in red:
        nid = node_id(test, is_file)
        if nid is None:
            unmapped.append(test)
        else:
            ids.append(nid)
    return {
        "count": len(red),
        "rerun": bool(ids),
        "mass_failure": False,
        "ids": sorted(set(ids)),
        "unmapped": unmapped,
    }


def github_output_lines(plan: dict[str, Any]) -> list[str]:
    return [
        f"count={plan['count']}",
        f"rerun={'true' if plan['rerun'] else 'false'}",
        f"mass_failure={'true' if plan['mass_failure'] else 'false'}",
    ]


# ------------------------------------------------------------------ verdict

def leg_status(rc: int, outcomes: dict[str, str], confirmed: list[str]) -> str:
    if confirmed:
        return RED
    if rc not in _TEST_RC or (rc == 1 and not red_ids(outcomes)) or not outcomes:
        return BROKEN
    return GREEN


def classify_leg(
    rc: int,
    outcomes: dict[str, str],
    rerun: dict[str, str] | None,
    max_ids: int = MASS_FAILURE_THRESHOLD,
    rerun_rc: int | None = None,
) -> dict[str, Any]:
    """One leg's verdict from the full run's outcomes and the rerun's. A
    `rerun_rc` other than 0 or 1 voids the rerun."""
    red = red_ids(outcomes)
    mass = len(red) > max_ids
    usable = rerun_rc is None or rerun_rc in _TEST_RC
    if mass or rerun is None or not usable:
        flaky: list[str] = []
    else:
        flaky = [t for t in red if rerun.get(t) == tor.PASSED]
    confirmed = [t for t in red if t not in flaky]
    return {
        "rc": rc,
        "red": red,
        "flaky": flaky,
        "confirmed": confirmed,
        "mass_failure": mass,
        "status": leg_status(rc, outcomes, confirmed),
        "rerun_rc": rerun_rc,
        "rerun_note": None if usable else f"the rerun exited {rerun_rc}, so every red test stays confirmed",
    }


# ------------------------------------------------------------------- record

def parse_leg_spec(value: str) -> tuple[str, str, str]:
    """`LEG=DIR[:ROLE]` as (leg, dir, role). A `:` suffix that is not a role
    is part of the directory."""
    leg, sep, rest = value.partition("=")
    if not sep or not leg or not rest:
        raise ValueError(f"expected LEG=DIR[:ROLE], got {value!r}")
    head, colon, tail = rest.rpartition(":")
    if colon and head and tail in ROLES:
        return leg, head, tail
    return leg, rest, CANARY


def bundle_attempt(name: str, leg: str, sha: str) -> int | None:
    """N for an artifact directory `nightly-<leg>-<sha>-a<N>`, else None."""
    m = re.fullmatch(re.escape(f"nightly-{leg}-{sha}-a") + r"(\d+)", name)
    return int(m.group(1)) if m else None


def newest_bundle(names: Iterable[str], leg: str, sha: str) -> str | None:
    """The artifact directory of the leg's highest attempt, or None."""
    best: tuple[int, str] | None = None
    for name in names:
        n = bundle_attempt(name, leg, sha)
        if n is not None and (best is None or n > best[0]):
            best = (n, name)
    return None if best is None else best[1]


def night_verdict(
    roles: dict[str, str], status: dict[str, str], required: Iterable[str] = DEFAULT_CANARY_LEGS,
) -> str:
    canary = [leg for leg, role in roles.items() if role == CANARY]
    if not canary or any(roles.get(leg) != CANARY for leg in required):
        return INCOMPLETE
    if any(status.get(leg, MISSING) == MISSING for leg in canary):
        return INCOMPLETE
    if all(status[leg] == GREEN for leg in canary):
        return GREEN
    return RED


def _is_str_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)  # pyright: ignore[reportUnknownVariableType]


def check_verdict(verdict: dict[str, Any], outcomes: dict[str, str]) -> str | None:
    """Why `verdict` cannot be trusted over the report it came from, or None."""
    rc = verdict.get("rc")
    if not isinstance(rc, int) or isinstance(rc, bool):
        return "verdict.json holds no pytest rc"
    if not all(_is_str_list(verdict.get(key)) for key in ("red", "flaky", "confirmed")):
        return "verdict.json does not list the red, flaky and confirmed tests"
    if not isinstance(verdict.get("mass_failure"), bool):
        return "verdict.json does not say whether this was a mass failure"
    red = red_ids(outcomes)
    if sorted(verdict["red"]) != red or sorted(verdict["flaky"] + verdict["confirmed"]) != red:
        return "verdict.json does not name the red tests in junit.xml"
    if verdict.get("status") != leg_status(rc, outcomes, verdict["confirmed"]):
        return f"verdict.json says {verdict.get('status')!r}, which junit.xml and its rc do not support"
    return None


def build_nightly_record(
    junit: dict[str, str | None],
    verdicts: dict[str, dict[str, Any] | None],
    envs: dict[str, dict[str, Any] | None],
    roles: dict[str, str],
    attempts: dict[str, int | None],
    sha: str,
    run_id: int,
    run_attempt: int,
    event: str,
    created_at: str,
    bundle_notes: dict[str, list[str]] | None = None,
    required: Iterable[str] = DEFAULT_CANARY_LEGS,
) -> dict[str, Any]:
    """The format-1 record for one night with its additive `nightly` key. A
    leg in `required` that the other arguments do not name reads as missing."""
    required = list(required)
    unnamed = [leg for leg in required if leg not in roles]
    roles = {**roles, **{leg: CANARY for leg in unnamed}}
    junit = {**junit, **{leg: None for leg in unnamed}}
    envs = {**envs, **{leg: None for leg in unnamed}}
    attempts = {**attempts, **{leg: None for leg in unnamed}}
    bundle_notes = {**(bundle_notes or {}), **{leg: ["no --leg names this canary leg"] for leg in unnamed}}
    record = tor.build_record(junit, sha)
    status: dict[str, str] = {}
    flaky: dict[str, list[str]] = {}
    confirmed: dict[str, list[str]] = {}
    rc: dict[str, int | None] = {}
    mass: dict[str, bool] = {}
    notes: dict[str, list[str]] = {}
    for leg in sorted(roles):
        v = verdicts.get(leg)
        notes[leg] = list(bundle_notes.get(leg, []))
        if v and v.get("rerun_note"):
            notes[leg].append(str(v["rerun_note"]))
        if junit.get(leg) is None or v is None:
            status[leg] = MISSING
            flaky[leg] = []
            confirmed[leg] = []
            rc[leg] = None if v is None else int(v["rc"])
            mass[leg] = False
            continue
        status[leg] = str(v["status"])
        flaky[leg] = sorted(v["flaky"])
        confirmed[leg] = sorted(v["confirmed"])
        rc[leg] = int(v["rc"])
        mass[leg] = bool(v["mass_failure"])
    record["nightly"] = {
        "run_id": run_id,
        "run_attempt": run_attempt,
        "event": event,
        "created_at": created_at,
        "roles": dict(sorted(roles.items())),
        "attempts": dict(sorted(attempts.items())),
        "env": dict(sorted(envs.items())),
        "rc": rc,
        "status": status,
        "mass_failure": mass,
        "notes": notes,
        "flaky": flaky,
        "confirmed": confirmed,
        "verdict": night_verdict(roles, status, required),
    }
    return record


# -------------------------------------------------------------- regressions

def canary_legs_of(record: dict[str, Any], listed: Iterable[str]) -> list[str]:
    """The listed legs the record does not mark `extra`, in order."""
    roles = record.get("nightly", {}).get("roles", {})
    return [leg for leg in listed if roles.get(leg, CANARY) == CANARY]


def baseline_outcome(last_green: dict[str, Any], leg: str, test: str) -> str | None:
    """A test's outcome on a leg of the last green record; a failure that
    was flaky there counts as `passed`. None when the record lacks it."""
    legs: dict[str, Any] = last_green.get("legs") or {}
    leg_record: dict[str, Any] | None = legs.get(leg)
    if not leg_record:
        return None
    tests: dict[str, str] = leg_record.get("tests", {})
    flaky: list[str] = last_green.get("nightly", {}).get("flaky", {}).get(leg, [])
    outcome = tests.get(test)
    if outcome in RED_OUTCOMES and test in flaky:
        return tor.PASSED
    return outcome


def even_sample(items: list[str], n: int) -> list[str]:
    """`n` items spread evenly over the sorted list, or all of them."""
    ordered = sorted(items)
    if len(ordered) <= n:
        return ordered
    return [ordered[i * len(ordered) // n] for i in range(n)]


def _version_key(leg: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", leg))


def find_regressions(
    current: dict[str, Any],
    last_green: dict[str, Any] | None,
    last_green_meta: dict[str, Any] | None,
    listed_legs: Iterable[str],
) -> dict[str, Any]:
    """The triage input for one night. Only canary legs are read."""
    nightly = current.get("nightly", {})
    sha = str(current.get("sha") or "")
    legs = canary_legs_of(current, listed_legs)
    status = nightly.get("status", {})
    confirmed_by_leg = {leg: list(nightly.get("confirmed", {}).get(leg, [])) for leg in legs}
    triage: dict[str, Any] = {
        "format": TRIAGE_FORMAT,
        "sha": sha,
        "run_id": nightly.get("run_id"),
        "verdict": nightly.get("verdict"),
        "canary_legs": legs,
        "last_green": last_green_meta,
        "mass_failure": any(nightly.get("mass_failure", {}).get(leg, False) for leg in legs),
        "events": [],
        "candidates": [],
        "new_failing": [],
        "drift_check": None,
    }
    broken = [leg for leg in legs if status.get(leg, MISSING) in (MISSING, BROKEN)]
    if broken:
        triage["events"].append({"key": "canary-broken", "kind": "canary-broken", "legs": broken, "tests": []})
    tests = sorted({t for ids in confirmed_by_leg.values() for t in ids})
    if not tests:
        return triage
    failing_legs = sorted(leg for leg, ids in confirmed_by_leg.items() if ids)
    if last_green is None or last_green_meta is None:
        triage["events"].append({
            "key": f"no-baseline-{sha[:12]}", "kind": "no-baseline", "legs": failing_legs, "tests": tests,
        })
        return triage
    green_sha = str(last_green_meta.get("sha") or "")
    if green_sha == sha:
        triage["events"].append({
            "key": f"drift-{green_sha[:12]}", "kind": "drift", "legs": failing_legs, "tests": tests,
        })
        return triage
    # `diff_records` names each red test with its outcome on the baseline;
    # the flaky rule above then reads a flaky failure there as a pass.
    red_before: dict[tuple[str, str], str | None] = {}
    for d in tor.diff_records(current, last_green):
        if d.leg in legs:
            for t in d.red:
                red_before[(d.leg, t.test)] = t.before
    candidates: dict[str, dict[str, Any]] = {}
    new_failing: dict[str, list[str]] = {}
    for leg in legs:
        for test in confirmed_by_leg[leg]:
            before = red_before.get((leg, test))
            if before is not None:
                before = baseline_outcome(last_green, leg, test)
            if before is None:
                new_failing.setdefault(test, []).append(leg)
            else:
                entry = candidates.setdefault(test, {"test": test, "legs": [], "before": {}})
                entry["legs"].append(leg)
                entry["before"][leg] = before
    triage["candidates"] = [candidates[t] for t in sorted(candidates)]
    triage["new_failing"] = [{"test": t, "legs": new_failing[t]} for t in sorted(new_failing)]
    if candidates:
        cand_legs = sorted({leg for c in candidates.values() for leg in c["legs"]}, key=_version_key)
        ids = sorted(candidates)
        if triage["mass_failure"]:
            ids = even_sample(ids, MASS_FAILURE_SAMPLE)
        triage["drift_check"] = {"sha": green_sha, "python": cand_legs[-1], "tests": ids}
    return triage


# ------------------------------------------------------------------ publish

def run_date(created_at: str) -> str:
    date = created_at[:10]
    if not _DATE.fullmatch(date):
        raise ValueError(f"created_at {created_at!r} does not start with YYYY-MM-DD")
    return date


def run_path(record: dict[str, Any]) -> str:
    nightly = record["nightly"]
    date = run_date(str(nightly["created_at"]))
    return f"runs/{date}-{int(nightly['run_id'])}-a{int(nightly['run_attempt'])}"


def index_entry(record: dict[str, Any], triage: dict[str, Any] | None) -> dict[str, Any]:
    nightly = record["nightly"]
    legs = record.get("legs", {})
    return {
        "run_id": int(nightly["run_id"]),
        "run_attempt": int(nightly["run_attempt"]),
        "date": run_date(str(nightly["created_at"])),
        "path": run_path(record),
        "sha": record.get("sha"),
        "verdict": nightly["verdict"],
        "status": nightly["status"],
        "counts": {leg: (None if data is None else data["counts"]) for leg, data in sorted(legs.items())},
        "flaky": nightly["flaky"],
        "events": [] if triage is None else [str(e["key"]) for e in triage.get("events", [])],
    }


def merge_index(lines: list[str], entry: dict[str, Any]) -> list[str]:
    """The index with one line per run: an older attempt's line of
    `entry`'s run is replaced by `entry`, last; a newer one is kept."""
    kept: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        data = json.loads(line)
        if int(data["run_id"]) != entry["run_id"]:
            kept.append(line)
        elif int(data["run_attempt"]) > entry["run_attempt"]:
            return [ln for ln in lines if ln.strip()]
    return [*kept, json.dumps(entry, sort_keys=True, separators=(",", ":"))]


def _run_order(pointer: dict[str, Any]) -> tuple[int, int]:
    return int(pointer["run_id"]), int(pointer.get("attempt", 0))


def _moved(
    old: dict[str, Any] | None,
    new: dict[str, Any],
    is_ancestor: Callable[[str, str], bool | None],
    on_main: Callable[[str], bool | None],
    what: str,
    notes: list[str],
    problems: list[str],
) -> dict[str, Any] | None:
    """The entry after a green night at `new`: `new` when it is on main and
    `old` is absent, off main, or an ancestor of it; otherwise `old`."""
    new_sha = str(new["sha"])
    answer = on_main(new_sha)
    if answer is not True:
        why = "is not on main" if answer is False else "could not be checked against main"
        problems.append(f"{what}: refused {new_sha[:12]}, which {why}")
        return old
    if old is None:
        notes.append(f"last green {what}: set to {new_sha[:12]}")
        return new
    old_sha = str(old["sha"])
    # git just placed `new`, so it can read main: a commit it cannot place is
    # not on main either.
    if on_main(old_sha) is not True:
        notes.append(f"::warning::last green {what}: {old_sha[:12]} is not on main, re-pointed to {new_sha[:12]}")
        return new
    if old_sha == new_sha:
        if _run_order(new) > _run_order(old):
            notes.append(f"last green {what}: run {_run_order(old)} -> {_run_order(new)} at {new_sha[:12]}")
            return new
        notes.append(f"last green {what}: stays at run {_run_order(old)}, not older than {_run_order(new)}")
        return old
    answer = is_ancestor(old_sha, new_sha)
    if answer is True:
        notes.append(f"last green {what}: {old_sha[:12]} -> {new_sha[:12]}")
        return new
    if answer is False:
        notes.append(f"last green {what}: stays at {old_sha[:12]}, which is later than {new_sha[:12]}")
    else:
        problems.append(f"{what}: stays at {old_sha[:12]}, ancestry against {new_sha[:12]} unknown")
    return old


def pointed_at(green: dict[str, Any]) -> set[str]:
    """The run directories `last-green.json` names."""
    entries = [green.get("canary"), *green.get("legs", {}).values()]
    return {str(entry["path"]) for entry in entries if entry is not None}


def missing_records(green: dict[str, Any], has_record: Callable[[str], bool]) -> list[str]:
    """A problem per run directory `last-green.json` names with no record."""
    return [f"names {path}, which holds no record" for path in sorted(pointed_at(green)) if not has_record(path)]


def off_main_pointers(green: dict[str, Any], on_main: Callable[[str], bool | None]) -> list[str]:
    """A problem per entry of `last-green.json` whose commit is not on main."""
    entries = [("canary", green.get("canary")), *((f"leg {leg}", p) for leg, p in sorted(green.get("legs", {}).items()))]
    return [
        f"{what}: names {str(pointer['sha'])[:12]}, which is not on main; the next green night on main re-points it"
        for what, pointer in entries
        if pointer is not None and on_main(str(pointer["sha"])) is not True
    ]


def next_last_green(
    current: dict[str, Any] | None,
    record: dict[str, Any],
    is_ancestor: Callable[[str, str], bool | None],
    on_main: Callable[[str], bool | None],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """`last-green.json` after this night, a printable note per entry it
    touched, and a problem per entry that needs an operator."""
    nightly = record["nightly"]
    old = current or {}
    pointer = {
        "sha": str(record["sha"]),
        "path": run_path(record),
        "run_id": int(nightly["run_id"]),
        "attempt": int(nightly["run_attempt"]),
        "date": run_date(str(nightly["created_at"])),
    }
    notes: list[str] = []
    problems: list[str] = []
    canary = old.get("canary")
    if nightly["verdict"] == GREEN:
        canary = _moved(canary, pointer, is_ancestor, on_main, "canary", notes, problems)
    legs: dict[str, Any] = dict(old.get("legs") or {})
    for leg, status in sorted(nightly["status"].items()):
        if status == GREEN:
            moved = _moved(legs.get(leg), pointer, is_ancestor, on_main, f"leg {leg}", notes, problems)
            if moved is not None:
                legs[leg] = moved
    new_green = {"canary": canary, "legs": dict(sorted(legs.items()))}
    problems.extend(off_main_pointers(new_green, on_main))
    return new_green, notes, problems


# --------------------------------------------------------------------- I/O

def uv_version() -> str | None:
    try:
        r = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=False, timeout=_UV_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "sqlite_version": sqlite3.sqlite_version,
        "ImageOS": os.environ.get("ImageOS"),
        "ImageVersion": os.environ.get("ImageVersion"),
        "uv": uv_version(),
    }


def git_is_ancestor(repo: Path, old: str, new: str) -> bool | None:
    """`git merge-base --is-ancestor`: True, False, or None when git cannot
    tell (an unknown commit, a missing repository)."""
    try:
        r = subprocess.run(
            ["git", "merge-base", "--is-ancestor", old, new],
            cwd=repo, capture_output=True, check=False, timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {0: True, 1: False}.get(r.returncode)


def resolve_main_ref(repo: Path, requested: str | None) -> str | None:
    """The ref that names main in `repo`, or None when none resolves."""
    for ref in [requested] if requested else ["origin/main", "main"]:
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
                cwd=repo, capture_output=True, check=False, timeout=_GIT_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if r.returncode == 0:
            return ref
    return None


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data  # pyright: ignore[reportUnknownVariableType]


def _dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=1, sort_keys=True) + "\n"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_junit(path: Path) -> dict[str, str]:
    return parse_report(path.read_text(encoding="utf-8"))


def cmd_env(args: argparse.Namespace) -> int:
    text = _dumps(environment())
    if args.dry_run:
        print(f"dry run: would write {args.out}")
        sys.stdout.write(text)
        return 0
    _write(args.out, text)
    return 0


def cmd_failed_ids(args: argparse.Namespace) -> int:
    root: Path = args.repo_root
    if not root.is_dir():
        raise ValueError(f"--repo-root {root} is not a directory")
    plan = plan_rerun(_read_junit(args.junit), lambda rel: (root / rel).is_file(), args.max)
    text = "".join(f"{nid}\n" for nid in plan["ids"])
    for test in plan["unmapped"]:
        print(f"::notice::no node id for {test}; it is not rerun and stays confirmed", file=sys.stderr)
    if plan["mass_failure"]:
        print(f"::warning::{plan['count']} red tests, more than {args.max}: mass failure, no rerun", file=sys.stderr)
    outputs = github_output_lines(plan)
    exit_status = 0 if plan["rerun"] else 1
    if args.dry_run:
        if plan["rerun"]:
            print(f"dry run: would write {len(plan['ids'])} node id(s) to {args.out}")
        else:
            print(f"dry run: would write no ids and remove any {args.out}")
        sys.stdout.write(text)
        if args.github_output is not None:
            print(f"dry run: would append to {args.github_output}:")
        for line in outputs:
            print(line)
        return exit_status
    if plan["rerun"]:
        _write(args.out, text)
    else:
        args.out.unlink(missing_ok=True)
    for line in outputs:
        print(line)
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as fh:
            fh.write("".join(f"{line}\n" for line in outputs))
    return exit_status


def cmd_verdict(args: argparse.Namespace) -> int:
    if (args.rerun is None) != (args.rerun_rc is None):
        raise ValueError("--rerun and --rerun-rc go together")
    outcomes = _read_junit(args.junit)
    rerun: dict[str, str] | None = None
    unreadable: str | None = None
    if args.rerun is not None:
        try:
            rerun = _read_junit(args.rerun)
        except (OSError, ValueError, ElementTree.ParseError) as exc:
            unreadable = f"the rerun left no readable report ({exc}), so every red test stays confirmed"
    verdict = classify_leg(args.rc, outcomes, rerun, args.max, args.rerun_rc)
    if unreadable is not None:
        verdict["rerun_note"] = unreadable
    print(
        f"pytest rc {verdict['rc']}: {len(verdict['red'])} red, {len(verdict['flaky'])} flaky, "
        f"{len(verdict['confirmed'])} confirmed; leg {verdict['status']}"
        + (" (mass failure, no rerun)" if verdict["mass_failure"] else "")
    )
    if verdict["rerun_note"]:
        print(f"::warning::{verdict['rerun_note']}")
    for test in verdict["flaky"]:
        print(f"flaky: {test}")
    for test in verdict["confirmed"]:
        print(f"confirmed: {test}")
    if args.out is not None:
        if args.dry_run:
            print(f"dry run: would write {args.out}")
        else:
            _write(args.out, _dumps(verdict))
    return 0 if verdict["status"] == GREEN else 1


def _load_bundle(base: Path, leg: str, sha: str) -> tuple[Path | None, int | None]:
    """The bundle directory for a leg under `base`, and its attempt."""
    if any((base / name).is_file() for name in _BUNDLE_FILES):
        return base, None
    if not base.is_dir():
        return None, None
    name = newest_bundle((p.name for p in base.iterdir() if p.is_dir()), leg, sha)
    if name is None:
        return None, None
    return base / name, bundle_attempt(name, leg, sha)


def _load_leg(bundle: Path) -> tuple[str | None, dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    """A bundle's JUnit text, verdict and environment, and a note per
    problem. A verdict comes back only when the leg can be trusted."""
    unreadable: tuple[type[Exception], ...] = (OSError, ValueError, ElementTree.ParseError)
    notes: list[str] = []
    text: str | None = None
    outcomes: dict[str, str] | None = None
    try:
        raw = (bundle / "junit.xml").read_text(encoding="utf-8")
        outcomes = parse_report(raw)
        text = raw
    except unreadable as exc:
        notes.append(f"junit.xml is unreadable: {exc}")
    verdict: dict[str, Any] | None = None
    try:
        verdict = _read_json(bundle / "verdict.json")
    except unreadable as exc:
        notes.append(f"verdict.json is unreadable: {exc}")
    env: dict[str, Any] | None = None
    env_ok = True
    if (bundle / "env.json").exists():
        try:
            env = _read_json(bundle / "env.json")
        except unreadable as exc:
            env_ok = False
            notes.append(f"env.json is unreadable: {exc}")
    if verdict is not None and outcomes is not None:
        problem = check_verdict(verdict, outcomes)
        if problem is not None:
            notes.append(problem)
            verdict = None
    if outcomes is None or not env_ok:
        verdict = None
    return text, verdict, env, notes


def cmd_record(args: argparse.Namespace) -> int:
    specs = [parse_leg_spec(v) for v in args.leg]
    seen = [leg for leg, _, _ in specs]
    if len(set(seen)) != len(seen):
        raise ValueError(f"a leg is named twice: {seen}")
    junit: dict[str, str | None] = {}
    verdicts: dict[str, dict[str, Any] | None] = {}
    envs: dict[str, dict[str, Any] | None] = {}
    roles: dict[str, str] = {}
    attempts: dict[str, int | None] = {}
    notes: dict[str, list[str]] = {}
    for leg, directory, role in specs:
        roles[leg] = role
        bundle, attempt = _load_bundle(Path(directory), leg, args.sha)
        attempts[leg] = attempt
        if bundle is None:
            junit[leg], verdicts[leg], envs[leg] = None, None, None
            notes[leg] = [f"no bundle under {directory}"]
        else:
            junit[leg], verdicts[leg], envs[leg], notes[leg] = _load_leg(bundle)
        for note in notes[leg]:
            print(f"::notice::leg {leg}: {note}", file=sys.stderr)
    created_at = args.created_at or datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    run_date(created_at)
    event = args.event if args.event is not None else os.environ.get("GITHUB_EVENT_NAME", "")
    record = build_nightly_record(
        junit, verdicts, envs, roles, attempts, args.sha, args.run_id, args.run_attempt, event, created_at, notes,
        [leg for leg in args.canary_legs.split(",") if leg],
    )
    nightly = record["nightly"]
    if args.dry_run:
        print(f"dry run: would write {args.out}")
    for leg in sorted(nightly["roles"]):
        print(
            f"leg {leg} ({nightly['roles'][leg]}): {nightly['status'][leg]}, "
            f"{len(nightly['flaky'][leg])} flaky, {len(nightly['confirmed'][leg])} confirmed"
        )
    print(f"night: {nightly['verdict']}")
    if not args.dry_run:
        _write(args.out, _dumps(record))
    return 0


def _last_green(directory: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """The last green record and its pointer, or (None, None)."""
    pointer_file = directory / "last-green.json"
    if not pointer_file.is_file():
        return None, None
    meta = _read_json(pointer_file).get("canary")
    if not meta:
        return None, None
    return _read_json(directory / str(meta["path"]) / "record.json"), dict(meta)


def cmd_regressions(args: argparse.Namespace) -> int:
    current = _read_json(args.current)
    last_green, meta = _last_green(args.last_green_dir)
    listed = [leg for leg in args.canary_legs.split(",") if leg]
    triage = find_regressions(current, last_green, meta, listed)
    if meta is None:
        print("no last green night yet")
    else:
        print(f"last green: {str(meta['sha'])[:12]} ({meta['path']})")
    for e in triage["events"]:
        print(f"event {e['key']}: {len(e['tests'])} test(s)")
    print(f"{len(triage['candidates'])} candidate(s), {len(triage['new_failing'])} new failing")
    if args.dry_run:
        print(f"dry run: would write {args.out}")
        return 0
    _write(args.out, _dumps(triage))
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    record = _read_json(args.record)
    if "nightly" not in record:
        raise ValueError(f"{args.record} has no nightly key")
    triage: dict[str, Any] | None = None
    if args.triage.is_file():
        triage = _read_json(args.triage)
    else:
        print(f"::notice::no triage at {args.triage}; publishing the record without event keys", file=sys.stderr)
    branch: Path = args.branch_dir
    if not branch.is_dir():
        raise ValueError(f"--branch-dir {branch} is not a directory")
    path = run_path(record)
    index_file = branch / "index.jsonl"
    lines = index_file.read_text(encoding="utf-8").splitlines() if index_file.is_file() else []
    index = merge_index(lines, index_entry(record, triage))
    green_file = branch / "last-green.json"
    old_green = _read_json(green_file) if green_file.is_file() else None
    main_ref = resolve_main_ref(args.main_repo, args.main_ref)
    placed: dict[str, bool | None] = {}

    def on_main(sha: str) -> bool | None:
        if sha not in placed:
            placed[sha] = None if main_ref is None else git_is_ancestor(args.main_repo, sha, main_ref)
        return placed[sha]

    new_green, notes, problems = next_last_green(
        old_green, record, lambda old, new: git_is_ancestor(args.main_repo, old, new), on_main,
    )
    record_file = branch / path / "record.json"
    if (
        path in pointed_at(old_green or {})
        and record_file.is_file()
        and record_file.read_text(encoding="utf-8") != _dumps(record)
    ):
        print(f"::error::last green names {path}, so its record stays as it is; nothing was written")
        return 1
    writes: list[tuple[Path, str]] = [(record_file, _dumps(record))]
    if triage is not None:
        writes.append((branch / path / "triage.json", _dumps(triage)))
    writes.append((index_file, "".join(f"{line}\n" for line in index)))
    if new_green != old_green:
        writes.append((green_file, _dumps(new_green)))
    problems.extend(missing_records(new_green, lambda p: p == path or (branch / p / "record.json").is_file()))
    for note in notes:
        print(note)
    for target, text in writes:
        if args.dry_run:
            print(f"dry run: would write {target.relative_to(branch).as_posix()}")
        else:
            _write(target, text)
    for problem in problems:
        print(f"::error::last green {problem}")
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dry-run", action="store_true", help="write no file, change no git state, print what would happen")

    p = sub.add_parser("env", parents=[common], help="write this interpreter's environment")
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("failed-ids", parents=[common], help="write the node ids of a run's red tests")
    p.add_argument("--junit", required=True, type=Path)
    p.add_argument("--repo-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max", type=int, default=MASS_FAILURE_THRESHOLD)
    p.add_argument("--github-output", default=None, type=Path)
    p.set_defaults(func=cmd_failed_ids)

    p = sub.add_parser("verdict", parents=[common], help="sort one leg's red tests into flaky and confirmed")
    p.add_argument("--rc", required=True, type=int)
    p.add_argument("--junit", required=True, type=Path)
    p.add_argument("--rerun", default=None, type=Path)
    p.add_argument("--rerun-rc", default=None, type=int)
    p.add_argument("--max", type=int, default=MASS_FAILURE_THRESHOLD)
    p.add_argument("--out", default=None, type=Path)
    p.set_defaults(func=cmd_verdict)

    p = sub.add_parser("record", parents=[common], help="build the night's record from the leg bundles")
    p.add_argument("--leg", action="append", required=True, metavar="LEG=DIR[:ROLE]")
    p.add_argument("--sha", required=True)
    p.add_argument("--run-id", required=True, type=int)
    p.add_argument("--run-attempt", required=True, type=int)
    p.add_argument("--event", default=None)
    p.add_argument("--created-at", default=None)
    p.add_argument("--canary-legs", default=",".join(DEFAULT_CANARY_LEGS))
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("regressions", parents=[common], help="compare the canary legs with the last green night")
    p.add_argument("--current", required=True, type=Path)
    p.add_argument("--last-green-dir", required=True, type=Path)
    p.add_argument("--canary-legs", default=",".join(DEFAULT_CANARY_LEGS))
    p.add_argument("--out", required=True, type=Path)
    p.set_defaults(func=cmd_regressions)

    p = sub.add_parser("publish", parents=[common], help="write the night to a results-branch checkout")
    p.add_argument("--record", required=True, type=Path)
    p.add_argument("--triage", required=True, type=Path)
    p.add_argument("--branch-dir", required=True, type=Path)
    p.add_argument("--main-repo", default=Path("."), type=Path)
    p.add_argument("--main-ref", default=None)
    p.set_defaults(func=cmd_publish)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, KeyError, TypeError, ElementTree.ParseError) as exc:
        print(f"nightly_canary: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
