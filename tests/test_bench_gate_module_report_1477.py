"""The bench-gate summary reports per module, not just in total (#1477 AC3).

#1456 gave the tier an aggregate skip count, which is the right answer
for this repository: with no corpus at all, "36 skipped" is the whole
story. It is the wrong answer the moment a corpus exists. The corpus
covers a minority of the modules scaffolded under `tests/corpus/`, so a
lab-side run reports a healthy "N passed" while most of the tier skipped
for want of rows — the same misreading #1456 closed, one level in.

Driven against the hook directly with constructed reports rather than by
running the tier under a fixture corpus: the states that need pinning
include ones this repository cannot produce (a module present and
non-empty, so its gate actually executed), and a test that can only
assert the states available locally would pin exactly the case that was
never in doubt.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_conftest_under_test", _REPO / "tests" / "conftest.py"
)
assert _spec and _spec.loader
conftest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conftest)

CORPUS_ENV_VAR = conftest.CORPUS_ENV_VAR
BENCH_GATE_SKIP_REASON = conftest.BENCH_GATE_SKIP_REASON


class _Report:
    """A pytest report as `pytest_terminal_summary` reads one."""

    def __init__(self, reason: str = "", keywords: dict[str, int] | None = None):
        # pytest stores a skip's reason as the third element of a
        # (path, lineno, reason) tuple. Anything else has no reason.
        self.longrepr = ("f.py", 1, reason) if reason else None
        self.keywords = keywords or {}


class _Terminal:
    """Captures what the hook writes, in order."""

    def __init__(self, stats: dict[str, list[_Report]]):
        self.stats = stats
        self.lines: list[str] = []

    def write_sep(self, _char: str, title: str) -> None:
        self.lines.append(f"=== {title}")

    def write_line(self, line: str) -> None:
        self.lines.append(line)


def _summary(stats: dict[str, list[_Report]]) -> list[str]:
    term = _Terminal(stats)
    conftest.pytest_terminal_summary(term)
    return term.lines


def _module_skip(module: str, why: str) -> _Report:
    """The reason `load_corpus_module` actually writes.

    Built from the same f-string shape rather than quoted by hand, so a
    change to the message reddens this file instead of silently
    disabling the classifier that parses it.
    """
    root = Path("/corpus")
    if why == "missing":
        return _Report(f"corpus module {module!r} missing under {root}")
    return _Report(f"corpus module {module!r} empty under {root}")


def test_no_bench_gate_activity_prints_nothing() -> None:
    """An ordinary unit-test run must not grow a section."""
    assert _summary({"passed": [_Report()], "skipped": [_Report("unrelated")]}) == []


def test_the_absent_tier_still_reports_its_aggregate() -> None:
    """#1456's behaviour, unchanged — this repository's whole case."""
    lines = _summary({"skipped": [_Report(BENCH_GATE_SKIP_REASON)] * 36})

    assert lines[0].endswith("bench-gate tier")
    assert "36 bench-gate tests skipped" in lines[1]
    assert "did NOT run" in lines[1]
    assert CORPUS_ENV_VAR in lines[1]


def test_a_skipped_module_is_named_and_counted() -> None:
    """The report #1477 AC3 asks for, and the one #1456 cannot give.

    Without this the run below reads as "2 passed" — a green tier — when
    two of its three modules produced no verdict at all.
    """
    lines = _summary({
        "passed": [_Report(keywords={"bench_gated": 1})] * 2,
        "skipped": [
            _module_skip("dedup", "missing"),
            _module_skip("dedup", "missing"),
            _module_skip("sentiment", "empty"),
        ],
    })
    body = "\n".join(lines)

    assert "2 bench-gate tests executed" in body
    assert "'dedup': 2 test(s) skipped" in body
    assert "is missing" in body
    assert "'sentiment': 1 test(s) skipped" in body
    assert "is empty" in body
    assert "no verdict" in body


def test_missing_and_empty_are_not_merged() -> None:
    """They are different failures: no rows delivered vs a dead module."""
    lines = _summary({"skipped": [
        _module_skip("dedup", "missing"), _module_skip("dedup", "empty"),
    ]})
    body = "\n".join(lines)

    assert body.count("'dedup'") == 2, body
    assert "is missing" in body and "is empty" in body


def test_an_executed_gate_is_counted_off_the_marker() -> None:
    """The marker is the only place that signal survives to summary time.

    A bench-gated test that ran leaves a `passed` report indistinguishable
    from any other unless its keywords are read.
    """
    lines = _summary({
        "passed": [_Report(keywords={"bench_gated": 1}), _Report()],
        "failed": [_Report(keywords={"bench_gated": 1})],
    })

    assert any("2 bench-gate tests executed" in line for line in lines), lines


def test_an_unrelated_skip_inside_the_tier_is_not_folded_in() -> None:
    """Classification is by reason, not by which file the test lives in.

    Counting every skip in a bench-gated module would inflate the
    missing-rows figure with skips that have nothing to do with the
    corpus — and inflating it is the direction that makes the tier look
    more blocked than it is, so nobody would chase it.
    """
    lines = _summary({"skipped": [
        _module_skip("dedup", "empty"),
        _Report("needs a network connection"),
        _Report("requires Windows"),
    ]})
    body = "\n".join(lines)

    assert "'dedup': 1 test(s) skipped" in body
    assert "network" not in body and "Windows" not in body


@pytest.mark.parametrize("reason", ["", "corpus module missing under /x"])
def test_a_reasonless_or_unparseable_skip_is_ignored(reason: str) -> None:
    """No crash, and no phantom module named from a partial match."""
    assert _summary({"skipped": [_Report(reason)]}) == []
