"""#1351: importing the hook must not import the numeric stack.

Every hook fire is a fresh process. Before this, `import aelfrice.hook` pulled
numpy, scipy (153 submodules) and snowballstemmer at module load — ~105 ms of
scipy alone — through two chains:

    aelfrice.hook -> context_rebuilder -> query_understanding -> store_cache -> bm25 -> scipy.sparse
    aelfrice.hook -> retrieval -> graph_spectral -> scipy.sparse.linalg

Half of UserPromptSubmit fires never reach the L1 lane at all (the shape gate
skips system-generated and trivial prompts), and no `Stop` / `PreToolUse` /
`PostToolUse` / `PreCompact` / `SessionStart` fire runs it, so those processes
paid the whole cost for nothing.

The assertion is on the *module set*, not on wall-clock milliseconds. A timing
budget here would be a flake generator under CI contention — #1307 is the
precedent, where exactly that shape was diagnosed as a deadlock twice.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

# Roots that must not appear in `sys.modules` after importing the hook.
_NUMERIC_ROOTS = ("numpy", "scipy", "snowballstemmer")

_PROBE = """
import sys
import aelfrice.hook
roots = {m.split(".")[0] for m in sys.modules}
print(",".join(sorted(roots & {%s})))
""" % ", ".join(repr(r) for r in _NUMERIC_ROOTS)


def _probe(statement: str) -> str:
    """Run `statement` in a clean interpreter and return its stdout."""
    proc = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


@pytest.mark.timeout(30)
def test_importing_the_hook_does_not_import_the_numeric_stack() -> None:
    """The gate. Must run in a subprocess.

    An in-process assertion is worthless: by the time pytest reaches this
    module it has already imported most of the tree, so `sys.modules` would
    contain numpy either way and the test would pass on the unfixed code.
    """
    assert _probe(_PROBE) == "", (
        "importing aelfrice.hook pulled the numeric stack back into the import "
        "graph. Every hook process pays this, including the ones that never "
        "retrieve. Find the new eager edge with:\n"
        "  python -X importtime -c 'import aelfrice.hook'"
    )


@pytest.mark.timeout(30)
def test_the_probe_can_see_the_numeric_stack_when_it_is_there() -> None:
    """Guard the guard: a probe that can never observe numpy proves nothing.

    If `_NUMERIC_ROOTS` were misspelled, or the subprocess silently failed to
    import anything, the assertion above would pass vacuously and read exactly
    like a clean import graph.
    """
    observed = _probe(
        "import sys, numpy, scipy.sparse, snowballstemmer\n"
        'roots = {m.split(".")[0] for m in sys.modules}\n'
        'print(",".join(sorted(roots & {%s})))'
        % ", ".join(repr(r) for r in _NUMERIC_ROOTS)
    )
    assert observed == "numpy,scipy,snowballstemmer", observed


@pytest.mark.timeout(30)
def test_the_numeric_stack_still_loads_when_retrieval_runs() -> None:
    """The imports moved; they were not removed.

    `retrieve` must still reach the L1 lane. If a deferred import were dropped
    rather than moved, this fails with NameError rather than passing quietly.
    """
    observed = _probe(
        "import sys\n"
        "from aelfrice.retrieval import retrieve\n"
        "import aelfrice.bm25\n"
        'roots = {m.split(".")[0] for m in sys.modules}\n'
        'print(",".join(sorted(roots & {%s})))'
        % ", ".join(repr(r) for r in _NUMERIC_ROOTS)
    )
    assert observed == "numpy,scipy,snowballstemmer", observed
