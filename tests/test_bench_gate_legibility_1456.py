"""The bench-gate tier's skips must not read as passes (#1456, #1420 §3).

36 quality gates under `tests/bench_gate/` skip on every public CI run because
`AELFRICE_CORPUS_ROOT` points at a private corpus that this repository cannot
carry. That is the ratified disposition. The failure it creates is a reading
failure: a green `N passed, M skipped` tail gives a reader no way to tell that
the retrieval, compression and clustering quality gates were never executed.

So two things are pinned here — the skip reason names the deciding issue, and
the run states the count on its own line.

Both tests drive pytest in a subprocess and therefore carry an explicit
`@pytest.mark.timeout`. The project sets a global `timeout = 30` sized for a
dedicated CI runner; a subprocess-driven test inheriting it fails from machine
contention rather than from anything it asserts (#1472).
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.conftest import BENCH_GATE_SKIP_REASON, CORPUS_ENV_VAR


def test_the_skip_reason_names_the_deciding_issue() -> None:
    """A reader who hits the skip must be able to find out why it exists.

    Mutation that turns this red: drop the `#1420 §3` reference back to the
    bare "lab corpus absent" wording the reason carried before.
    """
    assert CORPUS_ENV_VAR in BENCH_GATE_SKIP_REASON
    assert "#1420" in BENCH_GATE_SKIP_REASON, (
        "the skip reason must cite the disposition that decided this tier "
        "runs lab-side, or the reader has nowhere to go"
    )
    assert "by design" in BENCH_GATE_SKIP_REASON, (
        "the reason must say the skip is intentional; an unexplained skip "
        "reads as a broken environment"
    )


@pytest.mark.timeout(120)
def test_a_run_with_no_corpus_states_the_bench_gate_count(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The summary line is the whole point of AC2 — assert it is emitted.

    Runs the real bench-gate directory in a subprocess with the corpus
    unset, which is exactly the public-CI condition, and asserts the tier
    line appears with a non-zero count.

    Mutation that turns this red: delete `pytest_terminal_summary` from
    `tests/conftest.py`. The tests still skip and the run is still green --
    which is precisely the state this issue exists to make visible.
    """
    env = {
        k: v
        for k, v in __import__("os").environ.items()
        if k != CORPUS_ENV_VAR
    }
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "tests/bench_gate/",
            "-q", "-p", "no:cacheprovider",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=110,
    )
    out = proc.stdout

    assert "bench-gate tier" in out, (
        "no bench-gate summary section in the run output; the tier's skips "
        f"are indistinguishable from any other skip.\n{out[-3000:]}"
    )
    assert "did NOT run" in out, (
        "the summary must say plainly that the gates did not run"
    )
    assert "#1420" in out, "the summary must cite the disposition"

    # A count, and a real one -- a line that says "0 bench-gate tests
    # skipped" would satisfy a substring check while telling the reader
    # nothing.
    import re

    m = re.search(r"(\d+) bench-gate tests skipped", out)
    assert m, f"no count in the summary line.\n{out[-2000:]}"
    assert int(m.group(1)) > 0, (
        "the summary reported zero skipped bench-gate tests while running "
        "the bench-gate directory with no corpus"
    )
