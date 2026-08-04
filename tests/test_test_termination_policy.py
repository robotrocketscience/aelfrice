"""Every test must have an exit — enforced, not documented.

A test may end because its condition is met, because it fails, or
because a timer expires. What it may never do is block forever. The
failure this guards against is not a slow test; it is a test that
prints a verdict and then hangs, because that costs a CI job's full
wall clock and buries the real cause above the hang.

The concrete incident: `tests/test_feedback_atomicity.py` raced eight
non-daemon threads through an unbounded `threading.Barrier`. A worker
that raised before reaching the barrier (concurrent store init can,
see #1310) left the other seven waiting on a quorum that could never
form. `pytest-timeout` fired and reported `1 failed ... in 6.10s` — and
then the interpreter blocked in `threading._shutdown()` joining those
seven, and the process had to be killed externally. A green-looking
summary followed by an unbounded hang.

`pytest-timeout` alone does not close this. It bounds the *test*; it
does not bound the *process*, because it cannot reap threads it did not
start. So the bound has to exist at each blocking call.

This module walks the AST of everything under `tests/` and fails on any
blocking call with no ceiling. It is intentionally mechanical: the rule
is checkable without judgement, so it cannot rot into advice.

Scope and its limits, stated rather than implied:

  * It checks `subprocess.*` calls and the `threading` / `queue`
    blocking primitives by NAME. A blocking call reached through an
    alias (``from subprocess import run``) or a wrapper is not seen.
    Both patterns are absent from `tests/` today — the first assertion
    below pins that, so introducing one fails here rather than silently
    widening the hole.
  * It does not attempt to prove that a bounded call is bounded
    *tightly enough*. A `timeout=` of 30 s and one of 30 000 s both
    pass. The claim is termination, not promptness.
  * It does not analyse loops. `while True` with an internal `break` is
    indistinguishable from a spin without executing it.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parent

SUBPROCESS_BLOCKING = frozenset(
    {"run", "Popen", "check_output", "check_call", "call"}
)
"""`subprocess` entry points that wait on a child process."""

_UNAMBIGUOUS_SYNC = frozenset({"wait", "join"})
"""Zero-argument `.wait()` / `.join()` — always a sync primitive.

Checked on receiver-name-independent grounds, because the receiver name
is exactly what a deadlocking test does not tell you: the shape that hung
in #1318 was `t.join()`, and `t` matches no hint. The zero-argument form
disambiguates without a name: `str.join` and `os.path.join` both take
exactly one positional argument, and `Path.joinpath` is a different
attribute, so `x.join()` with no args cannot be one of them.
"""

_HINTED_SYNC = frozenset({"acquire", "get"})
"""Blocking methods that are genuinely ambiguous by name alone.

`.get()` on a dict is ubiquitous and never blocks, so these two are
gated on a receiver-name hint. `.join()`/`.wait()` are not — see above.
"""

SYNC_BLOCKING = _UNAMBIGUOUS_SYNC | _HINTED_SYNC
"""Blocking methods of `threading` / `queue` primitives.

Checked only when called with no positional argument and no `timeout=`
keyword. `join(5)` and `get(True, 5)` are bounded and pass.
"""

_SYNC_HINTS = ("barrier", "thread", "lock", "queue", "event", "cond", "sem")
"""Receiver-name substrings that mark an ambiguous call as a primitive.

Applies to `_HINTED_SYNC` only. `t.join()` and `barrier.wait()` — the
two shapes that actually deadlocked — are both caught without it.
"""


def _iter_test_sources() -> list[Path]:
    return sorted(TESTS_ROOT.rglob("*.py"))


def _receiver_name(node: ast.Attribute) -> str:
    value = node.value
    if isinstance(value, ast.Name):
        return value.id.lower()
    if isinstance(value, ast.Attribute):
        return value.attr.lower()
    return ""


def _is_bounded(call: ast.Call) -> bool:
    return any(kw.arg == "timeout" for kw in call.keywords)


def _unbounded_calls(path: Path) -> list[tuple[int, str]]:
    """Return `[(lineno, desc)]` for unbounded blocking calls."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - a broken test file fails elsewhere
        return []

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue

        receiver = _receiver_name(func)
        if receiver == "subprocess" and func.attr in SUBPROCESS_BLOCKING:
            if not _is_bounded(node):
                found.append((node.lineno, f"subprocess.{func.attr}()"))
            continue

        if func.attr in SYNC_BLOCKING and not node.args:
            if func.attr in _HINTED_SYNC and not any(
                hint in receiver for hint in _SYNC_HINTS
            ):
                continue
            if not _is_bounded(node):
                found.append((node.lineno, f"{receiver}.{func.attr}()"))
    return found


def test_no_unbounded_blocking_calls_in_tests() -> None:
    """No test may wait on a child process or a thread without a ceiling.

    Terminates by construction: a bounded walk over a finite file list,
    pure computation, no subprocess and no waiting of its own.
    """
    offenders: list[str] = []
    for path in _iter_test_sources():
        if path == Path(__file__).resolve():
            continue
        for lineno, what in _unbounded_calls(path):
            rel = path.relative_to(TESTS_ROOT.parent)
            offenders.append(f"{rel}:{lineno}: {what}")

    assert not offenders, (
        "unbounded blocking call(s) in tests — every test must have an "
        "exit, so pass an explicit timeout=:\n  "
        + "\n  ".join(offenders)
    )


def test_blocking_calls_are_not_reached_through_aliases() -> None:
    """The name-based check above must not be silently bypassable.

    Two bypasses, both of which make a blocking call invisible to
    `_unbounded_calls`, which matches on the literal receiver name
    `subprocess`:

    * `from subprocess import run` — the call becomes a bare `run(...)`.
    * `import subprocess as sp` — the receiver becomes `sp`.

    No test does either today; this pins both, so the first one to try
    fails here instead of quietly opening a hole.

    Terminates by construction: pure AST walk, no I/O beyond reads.
    """
    aliased: list[str] = []
    for path in _iter_test_sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        # Hoisted out of the node loop deliberately: it is a per-file
        # fact, and computing it per AST node made this test ~6x slower
        # than its sibling over the identical file list (3.47s vs 0.57s
        # locally) and pushed it past the 5s default on CI — the exact
        # failure mode this module exists to prevent.
        rel = path.relative_to(TESTS_ROOT.parent)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "subprocess" and alias.asname:
                        aliased.append(
                            f"{rel}:{node.lineno}: import subprocess as "
                            f"{alias.asname}"
                        )
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "subprocess":
                continue
            for alias in node.names:
                if alias.name in SUBPROCESS_BLOCKING:
                    aliased.append(f"{rel}:{node.lineno}: from subprocess "
                                   f"import {alias.name}")
    assert not aliased, (
        "blocking subprocess entry point imported by name, which the "
        "termination check cannot see. Call it as `subprocess.<name>(...)` "
        "so the ceiling is enforceable:\n  " + "\n  ".join(aliased)
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import subprocess\nsubprocess.run(['x'])\n", 1),
        ("import subprocess\nsubprocess.run(['x'], timeout=5)\n", 0),
        ("subprocess.check_output(['x'])\n", 1),
        ("barrier.wait()\n", 1),
        ("barrier.wait(timeout=3)\n", 0),
        ("thread.join()\n", 1),
        ("thread.join(5)\n", 0),
        # Must NOT fire on the common non-blocking shapes:
        ("d.get('k')\n", 0),
        ("Path(a).joinpath(b)\n", 0),
        ("','.join(parts)\n", 0),
        ("config.get()\n", 0),
    ],
)
def test_detector_distinguishes(
    tmp_path: Path, source: str, expected: int
) -> None:
    """The detector itself is pinned, both directions.

    A checker that never fires would pass the repo trivially and give
    false assurance; one that fires on `','.join(...)` would be turned
    off within a day. Both arms are asserted.

    Terminates by construction: parses a string, no waiting.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(source, encoding="utf-8")
    assert len(_unbounded_calls(probe)) == expected, source
