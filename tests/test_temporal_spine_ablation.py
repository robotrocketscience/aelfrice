"""Unit tests for the #1064 ablation bench's G2 top-rank invariance logic.

Covers the pure accumulator math in ``benchmarks/temporal_spine_ablation.py``
that answers the flip gate's "no top-rank regression" question: the spine
lane appends after the ``[locked, l25, l1, hrr]`` core and before BFS, so the
core prefix must stay identical between the lane-off and lane-on arms, the
lane may insert only below the core, and any dropped baseline belief must be a
BFS-tail item (rank >= core), never a core one.

These tests use hand-built id lists + telemetry-derived core lengths — no
LoCoMo dataset, no store — so they run in the ordinary pytest matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so the harness can import benchmarks.*
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmarks.temporal_spine_ablation import (
    RankInvarianceAccumulator,
    _lcp_len,
)


def test_lcp_len_basics() -> None:
    assert _lcp_len([], []) == 0
    assert _lcp_len(["a"], []) == 0
    assert _lcp_len(["a", "b", "c"], ["a", "b", "z"]) == 2
    assert _lcp_len(["a", "b"], ["a", "b"]) == 2
    assert _lcp_len(["x"], ["y"]) == 0


def test_pure_append_is_invariant() -> None:
    """Spine appends its hits after an identical core: no regression."""
    acc = RankInvarianceAccumulator()
    core = ["c0", "c1", "c2"]  # locked+l25+l1+hrr
    baseline = list(core)
    spine = list(core) + ["s0", "s1"]  # lane inserts below core
    acc.add(baseline, spine, core_len_base=3, core_len_spine=3, n_spine_added=2)
    assert acc.n_questions == 1
    assert acc.head_invariant == 1
    assert acc.top_rank_displacements == 0
    assert acc.core_mismatch == 0
    assert acc.tail_evicted_total == 0
    assert acc.spine_contributed_questions == 1
    assert acc.spine_added_sum == 2
    assert acc.head_invariant_rate() == 1.0
    assert acc.passed() is True


def test_core_displacement_is_a_regression() -> None:
    """A core belief dropped when the lane turns on fails the gate."""
    acc = RankInvarianceAccumulator()
    baseline = ["c0", "c1", "c2"]
    spine = ["c0", "c2", "s0"]  # c1 (rank 1, inside core) vanished
    acc.add(baseline, spine, core_len_base=3, core_len_spine=3, n_spine_added=1)
    assert acc.top_rank_displacements == 1
    assert acc.head_invariant == 0
    assert acc.passed() is False


def test_core_reorder_is_a_regression() -> None:
    """Reordering the core prefix (no drop) still fails head invariance."""
    acc = RankInvarianceAccumulator()
    baseline = ["c0", "c1", "c2"]
    spine = ["c0", "c2", "c1", "s0"]  # c1/c2 swapped within the core
    acc.add(baseline, spine, core_len_base=3, core_len_spine=3, n_spine_added=1)
    assert acc.head_invariant == 0
    # no id was *dropped* (all three survive), so this is caught by the
    # head-prefix check, not the displacement counter
    assert acc.top_rank_displacements == 0
    assert acc.passed() is False


def test_bfs_tail_eviction_is_not_a_regression() -> None:
    """A dropped BFS-tail item (rank >= core) is by-design, still passes."""
    acc = RankInvarianceAccumulator()
    baseline = ["c0", "c1", "bfs0", "bfs1"]  # core=2, then two BFS items
    spine = ["c0", "c1", "s0", "s1"]  # spine spent the budget bfs0/bfs1 had
    acc.add(baseline, spine, core_len_base=2, core_len_spine=2, n_spine_added=2)
    assert acc.top_rank_displacements == 0
    assert acc.head_invariant == 1
    assert acc.tail_eviction_questions == 1
    assert acc.tail_evicted_total == 2
    assert acc.passed() is True


def test_core_length_mismatch_fails() -> None:
    """If the lane changes the core length, the gate must not pass."""
    acc = RankInvarianceAccumulator()
    baseline = ["c0", "c1"]
    spine = ["c0", "c1", "s0"]
    acc.add(baseline, spine, core_len_base=2, core_len_spine=3, n_spine_added=1)
    assert acc.core_mismatch == 1
    assert acc.passed() is False


def test_empty_accumulator_does_not_pass() -> None:
    """Zero questions is not a pass — nothing was measured."""
    acc = RankInvarianceAccumulator()
    assert acc.n_questions == 0
    assert acc.head_invariant_rate() == 0.0
    assert acc.mean_lcp() == 0.0
    assert acc.passed() is False


def test_lcp_tracked_across_questions() -> None:
    acc = RankInvarianceAccumulator()
    acc.add(["a", "b", "c"], ["a", "b", "c", "s"], 3, 3, 1)  # lcp 3
    acc.add(["a", "b"], ["a", "z"], 1, 1, 0)  # lcp 1, core=1
    assert acc.min_lcp == 1
    assert acc.mean_lcp() == 2.0
    assert acc.spine_contributed_questions == 1  # only the first added spine
