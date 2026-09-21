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
from collections.abc import Sequence
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
CORPUS_SCHEMA_PROPERTY = conftest.CORPUS_SCHEMA_PROPERTY


class _Report:
    """A pytest report as `pytest_terminal_summary` reads one."""

    def __init__(
        self,
        reason: str = "",
        keywords: dict[str, int] | None = None,
        user_properties: Sequence[tuple[str, object]] | None = None,
    ):
        # pytest stores a skip's reason as the third element of a
        # (path, lineno, reason) tuple. Anything else has no reason.
        self.longrepr = ("f.py", 1, reason) if reason else None
        self.keywords = keywords or {}
        # Normalised to a tuple so either call shape works: the #1580
        # schema tests pass a tuple of pairs, the #1581 verdict helpers
        # a list of them, and the hook only ever iterates it.
        self.user_properties = tuple(user_properties or ())


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


def test_a_corpus_schema_skip_is_not_counted_as_a_tier_skip() -> None:
    """Naming the env var is not the same as being a bench gate (#1580).

    The corpus-schema walk names `AELFRICE_CORPUS_ROOT` in its own skip
    reasons, one per scaffolded module, and it is not part of the
    retrieval / compression / clustering tier. Classifying on the
    substring folded all of them into the tier's figure, which is the
    direction that overstates how blocked the tier is.
    """
    schema_skip = _Report(
        f"corpus-schema: module 'dedup' holds no JSONL rows under /corpus "
        f"(root resolved from ${CORPUS_ENV_VAR}). Nothing was validated."
    )
    assert _summary({"skipped": [schema_skip] * 19}) == []

    lines = _summary({"skipped": [schema_skip, _Report(BENCH_GATE_SKIP_REASON)]})
    assert "1 bench-gate tests skipped" in "\n".join(lines)


def test_the_corpus_schema_row_count_is_printed() -> None:
    """"0 rows validated" and "all rows valid" are the same green tail.

    They were indistinguishable for the life of `test_corpus_schema.py`
    (#1580), so the count is printed rather than inferred.
    """
    lines = _summary({
        "passed": [
            _Report(user_properties=((CORPUS_SCHEMA_PROPERTY, "module 'x': 7 rows"),))
        ]
    })

    assert lines[0].endswith("corpus schema")
    assert "module 'x': 7 rows" in lines[1]


def test_a_failing_corpus_schema_test_still_reports_its_count() -> None:
    """The run that most needs the number is the one that went red."""
    lines = _summary({
        "failed": [
            _Report(user_properties=((CORPUS_SCHEMA_PROPERTY, "module 'y': 0 rows"),))
        ]
    })

    assert any("module 'y': 0 rows" in line for line in lines), lines


def test_an_errored_corpus_schema_test_still_reports_its_count() -> None:
    """A setup or teardown error files under `error`, not `failed`.

    Reading only `passed` and `failed` dropped the count on the one
    outcome where nothing else says how many rows were read.
    """
    lines = _summary({
        "error": [
            _Report(user_properties=((CORPUS_SCHEMA_PROPERTY, "module 'z': 3 rows"),))
        ]
    })

    assert any("module 'z': 3 rows" in line for line in lines), lines


def test_the_same_corpus_schema_line_is_printed_once() -> None:
    """A teardown error files a second report for a test that passed."""
    prop = ((CORPUS_SCHEMA_PROPERTY, "module 'w': 5 rows"),)
    lines = _summary({
        "passed": [_Report(user_properties=prop)],
        "error": [_Report(user_properties=prop)],
    })

    assert sum("module 'w': 5 rows" in line for line in lines) == 1, lines


# ---------------------------------------------------------------------------
# #1581 — rejected-by-null-model, and the states that used to vanish
# ---------------------------------------------------------------------------

BENCH_NULL_VERDICT_PROPERTY = conftest.BENCH_NULL_VERDICT_PROPERTY


def _rejected(module: str, why: str) -> _Report:
    return _Report(
        keywords={"bench_gated": 1},
        user_properties=[(BENCH_NULL_VERDICT_PROPERTY, f"{module}|REJECT|{why}")],
    )


def test_a_rejected_corpus_is_its_own_state() -> None:
    """AC3. Rejected is neither executed nor skipped.

    A corpus its own null model defeats graded nothing, so reporting it
    under either of the other two states restates the defect #1581
    closes.
    """
    lines = _summary({
        "failed": [_rejected("query_strategy", "gold == pool on 30 of 30 rows")],
        "passed": [_Report(keywords={"bench_gated": 1})],
    })
    body = "\n".join(lines)

    assert "1 bench-gate tests executed" in body
    assert "1 corpus module(s) REJECTED" in body
    assert "'query_strategy': REJECTED" in body
    assert "gold == pool on 30 of 30 rows" in body


def test_a_rejected_gate_is_not_counted_as_executed() -> None:
    """The headline number is the one a release reviewer reads."""
    lines = _summary({"failed": [_rejected("query_strategy", "why")]})

    assert not any("tests executed" in line for line in lines), lines
    assert any("REJECTED" in line for line in lines), lines


def test_an_accepted_gate_still_counts_as_executed() -> None:
    """An ACCEPT verdict must not be misread as a rejection."""
    lines = _summary({"passed": [_Report(
        keywords={"bench_gated": 1},
        user_properties=[(BENCH_NULL_VERDICT_PROPERTY, "sentiment|ACCEPT|")],
    )]})

    assert any("1 bench-gate tests executed" in line for line in lines), lines


def test_an_underfilled_module_reports_as_unverified() -> None:
    """AC5. The state that printed nothing at all before #1581.

    `_MODULE_SKIP_RE` matched only `missing|empty`, so a module skipped
    for being under its row floor produced neither an "executed" line
    nor a "no verdict" line, and a reader counting modules never saw it.
    """
    lines = _summary({"skipped": [
        _Report("corpus module 'directive_detection' underfilled under /c: 29 rows < 200 floor"),
    ]})
    body = "\n".join(lines)

    assert "'directive_detection': 1 test(s) skipped" in body
    assert "UNVERIFIED" in body


def test_an_unknown_skip_state_still_reports() -> None:
    """A state nobody added a sentence for must not vanish silently."""
    lines = _summary({"skipped": [
        _Report("corpus module 'sentiment' quarantined under /c"),
    ]})
    body = "\n".join(lines)

    assert "'sentiment': 1 test(s) skipped" in body
    assert "UNVERIFIED" in body
