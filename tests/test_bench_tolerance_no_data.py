"""The band-check must fail closed when nothing was measured (#1160).

Ignoring an individual SKIP leaf is correct and ratified (#479): one
uncomputable metric is not a regression. But the rollup ignored *every*
SKIP, so a run in which no metric could be computed — every adapter
exiting because its data dir was absent, which is what a failed dataset
download on the runner looks like — rolled up to PASS, and
`bench-canonical.yml` exited 0. The nightly reported success having
measured nothing. An empty check list did the same.

PASS now requires at least one leaf that actually passed; otherwise the
rollup is NO_DATA, which the workflow treats as failing. #479's case is
unchanged, and is re-pinned here rather than only in
`test_bench_tolerance_skip.py` because it is the invariant this change
could most easily have broken.
"""
from __future__ import annotations

import re
from pathlib import Path

from benchmarks.tolerance import BandCheck, Verdict, summarize

_WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "bench-canonical.yml"
)


def _leaf(name: str, verdict: Verdict) -> BandCheck:
    return BandCheck(
        path=(name,),
        canonical=0.5,
        observed=0.5,
        lower=0.48,
        upper=0.52,
        band_kind="relative",
        verdict=verdict,
        note="",
    )


def test_an_all_skip_run_is_no_data_not_pass() -> None:
    checks = [_leaf(f"m{i}", Verdict.SKIP) for i in range(12)]
    overall, counts = summarize(checks)
    assert overall == Verdict.NO_DATA
    assert counts[Verdict.SKIP.value] == 12
    assert counts[Verdict.PASS.value] == 0


def test_an_empty_check_list_is_no_data_not_pass() -> None:
    """The degenerate case: nothing to compare at all."""
    overall, counts = summarize([])
    assert overall == Verdict.NO_DATA
    assert all(count == 0 for count in counts.values())


def test_a_skip_beside_a_real_pass_still_passes() -> None:
    """#479 preserved: partial data is not a failure."""
    overall, counts = summarize([_leaf("a", Verdict.SKIP), _leaf("b", Verdict.PASS)])
    assert overall == Verdict.PASS
    assert counts[Verdict.SKIP.value] == 1


def test_fail_and_warn_outrank_no_data() -> None:
    """A compared-and-out-of-band leaf means measurement did happen."""
    assert summarize([_leaf("a", Verdict.SKIP), _leaf("b", Verdict.FAIL)])[0] is Verdict.FAIL
    assert summarize([_leaf("a", Verdict.SKIP), _leaf("b", Verdict.WARN)])[0] is Verdict.WARN
    # WARN with no PASS at all is still WARN, not NO_DATA — the leaf was
    # compared to its band, so the run is not evidence-free.
    assert summarize([_leaf("b", Verdict.WARN)])[0] is Verdict.WARN


def test_no_data_is_rollup_only() -> None:
    """No leaf ever carries NO_DATA, so counts need no bucket for it."""
    _, counts = summarize([_leaf("a", Verdict.SKIP)])
    assert Verdict.NO_DATA.value not in counts


def test_an_unexpected_leaf_verdict_is_counted_not_raised() -> None:
    """The gate must report a surprise, not crash inside the reporter."""
    overall, counts = summarize([_leaf("a", Verdict.NO_DATA)])
    assert counts[Verdict.NO_DATA.value] == 1
    assert overall == Verdict.NO_DATA  # no PASS leaf


def test_the_workflow_fails_the_job_on_no_data() -> None:
    """Otherwise the new verdict is inert and the gate still fails open."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    exits = [
        line.strip()
        for line in text.splitlines()
        if re.search(r"sys\.exit\(", line)
    ]
    assert exits, f"no sys.exit(...) found in {_WORKFLOW.name}"
    assert all("no_data" in line for line in exits), (
        f"the band-check step exits without accounting for 'no_data', so an "
        f"all-SKIP run still reports success: {exits}"
    )
