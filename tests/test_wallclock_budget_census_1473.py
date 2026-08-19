"""Default-run wall-clock budgets stay enumerated and capped (#1473).

A test that reads a clock and asserts against a numeric literal fails as a
function of machine load, not of any regression. Seven gate runs were lost to
seven different such tests in one session, none related to the diff under test.

`#1472` fixed the dominant mechanism (the global 5s `pytest-timeout` under
load, now scaled per machine). This file guards the *other* class: assertions
that compare a measured duration against a literal on every default run.

## The enumeration rule, and why weaker ones failed

A test is in the population when it **reads a clock** — `time.time`,
`perf_counter`, `monotonic`, `process_time`, as an attribute *or* a bare
imported name — **and** contains an `assert` comparing against a **numeric
literal** with `<`, `<=`, `>` or `>=`.

Two earlier scans were wrong:

1. Matching identifiers that look temporal (`elapsed`, `duration`, `took`,
   `latency`) found 5 and missed `p95`, `median` and `build_time`.
   Percentiles, means and per-item averages launder the name away. The signal
   is the clock read, not the variable name.
2. Treating `--run-perf` opt-in as file-scoped wrongly exempts tests.
   `tests/test_bm25_index.py` has a `--run-perf` gated test *and* an ungated
   sibling, and the ungated one is the one that went red. Gating is a
   `pytest.skip()` inside the body, so it is not visible from decorators.

Class nesting matters too: an entry that drops its `TestX::` prefix is not a
runnable node id, which is how a census silently omits a method.

## What this test does and does not do

It caps the population at its measured size and names the members, so adding a
new default-run wall-clock budget fails here and has to be argued for. It does
**not** claim the listed tests are safe — the one that actually failed under
load, `test_dotdir_plan_scales_linearly`, was converted to a comparison count
in the same change and is no longer in this list.

There is a third class this cannot see at all: tests with no elapsed assertion
that are bounded only by the global timeout (a scrypt KDF derive, a cold
`scipy.linalg` import). No assert-shaped scan finds those. #1472 covers them.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TESTS = REPO / "tests"

_CLOCKS = frozenset(
    {
        "time",
        "perf_counter",
        "monotonic",
        "process_time",
        "perf_counter_ns",
        "monotonic_ns",
    }
)

# Measured 2026-08-19. Every entry is a default-run test that compares a
# measured duration against a literal. Adding one is a deliberate act: convert
# it to a counting assertion (see test_dotdir_plan_dedupes_without_rescanning)
# or gate it behind --run-perf, and only then change this list.
KNOWN: frozenset[str] = frozenset(
    {
        "tests/regression/test_onboard_perf_50k_loc.py::test_scan_repo_under_60s_on_50k_loc",
        "tests/test_bayesian_ranking.py::test_ac11_per_query_overhead_within_budget",
        "tests/test_bfs_multihop.py::test_ac3_cyclic_graph_terminates_deterministically",
        "tests/test_bfs_multihop.py::test_ac10_latency_band_1k_beliefs_5k_edges",
        "tests/test_bm25_index.py::test_build_under_1s_at_n10k",
        "tests/test_ingest_log.py::test_ingest_latency_within_budget",
        "tests/test_transcript_logger.py::test_per_turn_latency_under_budget",
        "tests/test_working_state.py::TestLatency::test_projector_under_500ms_on_clean_repo",
    }
)


class _Body(ast.NodeVisitor):
    """Does this function read a clock, assert on a literal, or self-skip?"""

    def __init__(self) -> None:
        self.reads_clock = False
        self.literal_compare = False
        self.self_skips = False

    def visit_Call(self, node: ast.Call) -> None:
        fn = node.func
        if isinstance(fn, ast.Attribute) and fn.attr in _CLOCKS:
            self.reads_clock = True
        if isinstance(fn, ast.Name) and fn.id in _CLOCKS:
            self.reads_clock = True
        # `pytest.skip(...)` in the body is how --run-perf gating is spelled.
        if isinstance(fn, ast.Attribute) and fn.attr == "skip":
            self.self_skips = True
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        for cmp in ast.walk(node.test):
            if not isinstance(cmp, ast.Compare):
                continue
            if not any(
                isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                for op in cmp.ops
            ):
                continue
            for operand in [cmp.left, *cmp.comparators]:
                if isinstance(operand, ast.Constant) and isinstance(
                    operand.value, (int, float)
                ):
                    self.literal_compare = True
        self.generic_visit(node)


def _walk(node: ast.AST, prefix: list[str]):
    """Yield (node-id suffix, function) with class nesting preserved."""
    for child in getattr(node, "body", []):
        if isinstance(child, ast.ClassDef):
            yield from _walk(child, [*prefix, child.name])
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if child.name.startswith("test_"):
                yield "::".join([*prefix, child.name]), child


def census() -> tuple[set[str], int]:
    """Return (default-run clock-asserting node ids, files scanned)."""
    found: set[str] = set()
    scanned = 0
    for path in sorted(TESTS.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        scanned += 1
        rel = path.relative_to(REPO).as_posix()
        for suffix, fn in _walk(tree, []):
            v = _Body()
            for stmt in fn.body:
                v.visit(stmt)
            if v.reads_clock and v.literal_compare and not v.self_skips:
                found.add(f"{rel}::{suffix}")
    return found, scanned


def test_the_census_is_not_vacuous() -> None:
    """A scan that reaches no files passes for the wrong reason."""
    _, scanned = census()
    assert scanned > 100, (
        f"only {scanned} test files scanned — the census is not reaching the "
        "suite it is supposed to enumerate"
    )


def test_no_new_default_run_wall_clock_budget() -> None:
    """The population may shrink freely. Growing it is a deliberate act."""
    found, _ = census()
    added = found - KNOWN
    assert not added, (
        "new default-run wall-clock budget(s):\n  "
        + "\n  ".join(sorted(added))
        + "\n\nThese fail from machine load, not from a regression. Convert to "
        "a counting assertion (see test_dotdir_plan_dedupes_without_rescanning "
        "for the pattern) or gate behind --run-perf. If the budget is genuinely "
        "wanted, add it to KNOWN in this file with the reason."
    )


def test_the_known_list_has_no_stale_entries() -> None:
    """A converted test left in KNOWN hides the next real addition."""
    found, _ = census()
    stale = KNOWN - found
    assert not stale, (
        "KNOWN names test(s) that are no longer default-run wall-clock "
        "budgets:\n  " + "\n  ".join(sorted(stale))
        + "\n\nRemove them, or the list stops being an accurate cap."
    )
