#!/usr/bin/env python3
"""Mutation check for the `aelf core` feedback-event gate: each mutant must fail a test.

Each mutant replaces one exact snippet of a source file the gate uses. The script
copies the tree into a temporary folder, applies one mutant at a time there, runs
the core tests against the copy, and puts the snippet back. Your checkout is never
changed.

Usage:
    uv run python scripts/mutants_core_feedback.py --dry-run      # check every anchor, run nothing
    uv run python scripts/mutants_core_feedback.py --list         # print each mutant
    uv run python scripts/mutants_core_feedback.py                # run every mutant
    uv run python scripts/mutants_core_feedback.py NAME [NAME...] # run the named mutants

It runs the tests on the unmutated copy first and exits 1 if they fail. Then it
prints one line per mutant, `killed NAME by TEST` or `SURVIVED NAME`, and exits 1
when a mutant survives or its anchor isn't found exactly once.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent.parent
MD = "src/aelfrice/models.py"
CLI = "src/aelfrice/cli.py"
ST = "src/aelfrice/store.py"
HK = "src/aelfrice/hook.py"
DR = "src/aelfrice/doctor.py"
TESTS = (
    "tests/test_core_feedback_gate.py",
    "tests/test_cli_core.py",
)
COPIED = ("src", "tests", "docs", "scripts", "benchmarks", "CHANGELOG", ".github")
COPIED_FILES = ("pyproject.toml", "README.md", "uv.lock")
TEST_TIMEOUT_SECONDS = 600

# The tests whose fixtures gained feedback rows for the gate (operator ruling
# 2026-11-09). The cadence mutants below show they still catch a broken batch.
CADENCE_TESTS = (
    "tests/test_core_gate_session_end_1638.py",
    "tests/test_core_gate_background_1741.py",
    "tests/test_core_gate_session_end_flush_1638.py",
)


class Mutant(NamedTuple):
    """One exact snippet of `path` and its replacement; the snippet must occur
    once. `tests` are the files run against it."""

    name: str
    path: str
    old: str
    new: str
    tests: tuple[str, ...] = TESTS


MUTANTS: list[Mutant] = [Mutant(*m) for m in [
    ("threshold_one", MD,
     "CORE_MIN_POSTERIOR_FEEDBACK_EVENTS: Final[int] = 2",
     "CORE_MIN_POSTERIOR_FEEDBACK_EVENTS: Final[int] = 1"),
    ("threshold_three", MD,
     "CORE_MIN_POSTERIOR_FEEDBACK_EVENTS: Final[int] = 2",
     "CORE_MIN_POSTERIOR_FEEDBACK_EVENTS: Final[int] = 3"),
    ("cli_gate_off", CLI,
     "    if feedback_events < min_events:\n        return False\n", ""),
    ("cli_gate_le", CLI,
     "    if feedback_events < min_events:", "    if feedback_events <= min_events:"),
    ("cli_select_ignores_count", CLI,
     "                    feedback_counts.get(b.id, 0),\n", "                    99,\n"),
    ("cli_label_ignores_count", CLI,
     "            if _posterior_signal(b, args, fc.get(b.id, 0)):  # type: ignore[attr-defined]\n                signals.append",
     "            if _posterior_signal(b, args, 99):  # type: ignore[attr-defined]\n                signals.append"),
    ("cli_text_ignores_count", CLI,
     "        if _posterior_signal(b, args, fc.get(b.id, 0)):  # type: ignore[attr-defined]\n            parts.append",
     "        if _posterior_signal(b, args, 99):  # type: ignore[attr-defined]\n            parts.append"),
    ("store_counts_zero_valence", ST,
     '"WHERE valence != 0 "', '"WHERE 1 = 1 "'),
    ("store_counts_exposure", ST,
     '"AND source NOT IN (SELECT je.value FROM json_each(?) je) "',
     '"AND ? IS NOT NULL "'),
    ("hook_gate_off", HK,
     "    if feedback_events < CORE_MIN_POSTERIOR_FEEDBACK_EVENTS:\n        return False\n", ""),
    ("hook_lane_ignores_count", HK,
     "            b, episodes.get(b.id, 0), feedback_counts.get(b.id, 0),\n        ):\n            core_candidates.append(b)",
     "            b, episodes.get(b.id, 0), 99,\n        ):\n            core_candidates.append(b)"),
    ("doctor_candidates_ignore_count", DR,
     "        if qualifies(b, episodes.get(bid, 0), feedback_counts.get(bid, 0)):\n            candidates.append(b)",
     "        if qualifies(b, episodes.get(bid, 0), 99):\n            candidates.append(b)"),
]] + [
    # A session's own open batch stops claiming its hashes: every Stop after
    # the first flush asks again.
    Mutant("session_batch_claims_nothing", ST,
           "(cb.origin = ? AND cb.session_id = ?)",
           "(cb.origin = ? AND cb.session_id = ? AND 0)", CADENCE_TESTS),
    # The session-end batch keeps one candidate fewer than MAX_BATCH.
    Mutant("session_batch_cap_off_by_one", HK,
           "        batch = candidates[:MAX_BATCH]\n",
           "        batch = candidates[:MAX_BATCH - 1]\n", CADENCE_TESTS),
]

# Mutants dropped from the set, with the reason. A test can't tell them from the code.
RETIRED: dict[str, str] = {}


def _check_anchors(root: Path) -> list[str]:
    """Return a line for each mutant whose snippet isn't in its file exactly once."""
    bad: list[str] = []
    for name, path, old, _new, _tests in MUTANTS:
        count = (root / path).read_text(encoding="utf-8").count(old)
        if count != 1:
            bad.append(f"NOTFOUND {name} {count}")
    return bad


def _copy_tree(into: Path) -> None:
    for name in COPIED:
        if (ROOT / name).exists():
            shutil.copytree(ROOT / name, into / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in COPIED_FILES:
        if (ROOT / name).exists():
            shutil.copy(ROOT / name, into / name)


def _pytest(root: Path, tests: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, PYTHONPATH=str(root / "src"), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(
        [sys.executable, "-m", "pytest", *tests, "-x", "-q", "--no-header",
         "-p", "no:cacheprovider", "--tb=no", "--timeout=240"],
        cwd=root, env=env, capture_output=True, text=True, timeout=TEST_TIMEOUT_SECONDS,
    )


def _baseline_passes(root: Path) -> bool:
    """True when the tests pass on the unmutated copy, so a failure means a kill."""
    tests = tuple(dict.fromkeys(t for m in MUTANTS for t in m.tests))
    try:
        result = _pytest(root, tests)
    except subprocess.TimeoutExpired:
        print("baseline timed out", flush=True)
        return False
    if result.returncode != 0:
        print(f"baseline failed rc={result.returncode}", flush=True)
        return False
    print("baseline passed", flush=True)
    return True


def _run_one(root: Path, m: Mutant) -> bool:
    """Apply one mutant in `root`, run its tests, restore the file. True when killed."""
    name = m.name
    target = root / m.path
    original = target.read_text(encoding="utf-8")
    target.write_text(original.replace(m.old, m.new), encoding="utf-8")
    try:
        result = _pytest(root, m.tests)
    except subprocess.TimeoutExpired:
        print(f"killed   {name} by timeout", flush=True)
        return True
    finally:
        target.write_text(original, encoding="utf-8")
    if result.returncode == 0:
        print(f"SURVIVED {name}", flush=True)
        return False
    failed = [line for line in result.stdout.splitlines() if line.startswith(("FAILED", "ERROR"))]
    first = failed[0].split("::")[-1].split(" - ")[0][:70] if failed else f"rc={result.returncode}"
    print(f"killed   {name} by {first}", flush=True)
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Mutation check for the core feedback-event gate.")
    parser.add_argument("names", nargs="*", help="run only these mutants")
    parser.add_argument("--dry-run", action="store_true", help="check every anchor and run nothing")
    parser.add_argument("--list", action="store_true", help="print each mutant and retired name")
    parser.add_argument("--shard", default="0/1", help="I/N: run mutants whose index mod N is I")
    args = parser.parse_args(argv)
    names = {m[0] for m in MUTANTS}
    if len(names) != len(MUTANTS):
        print("duplicate mutant names", file=sys.stderr)
        return 1
    unknown = sorted(set(args.names) - names)
    if unknown:
        print("unknown mutants: " + " ".join(unknown), file=sys.stderr)
        return 1
    if args.list:
        for m in MUTANTS:
            print(f"{m.name}\t{m.path}")
        for name, why in sorted(RETIRED.items()):
            print(f"{name}\tretired: {why}")
        return 0
    bad = _check_anchors(ROOT)
    for line in bad:
        print(line)
    if args.dry_run:
        print(f"checked {len(MUTANTS)} anchors, {len(bad)} not found exactly once")
        return 1 if bad else 0
    shard, shards = (int(part) for part in args.shard.split("/"))
    survivors = 0
    killed = 0
    with tempfile.TemporaryDirectory(prefix="mutants-core-feedback-") as folder:
        root = Path(folder)
        _copy_tree(root)
        if not _baseline_passes(root):
            return 1
        for number, m in enumerate(MUTANTS):
            if (args.names and m.name not in args.names) or number % shards != shard:
                continue
            if (root / m.path).read_text(encoding="utf-8").count(m.old) != 1:
                continue
            if _run_one(root, m):
                killed += 1
            else:
                survivors += 1
    print(f"killed {killed}, survived {survivors}, anchors not found {len(bad)}")
    return 1 if survivors or bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
