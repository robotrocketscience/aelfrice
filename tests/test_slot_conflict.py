"""Tests for _slot_conflict and _slot_conflict_preextracted (#938).

Covers:
- Same slot, same object → no flag
- Same slot, different object → flag
- Different slots entirely → no flag
- Empty locked set → no flag
- Multiple locked beliefs, first match wins
- Latency budget (median of 100 rounds of 10 hits × 5 locked ≤ 5 ms; budget
  is 2 ms per spec but relaxed to 5 ms to accommodate slower CI hardware)
"""
from __future__ import annotations

import statistics
import time

import pytest

from aelfrice import contradiction
from aelfrice.contradiction import _slot_conflict, _slot_conflict_preextracted
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.value_compare import SlotConflict, ValueSlots, extract_values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk(
    bid: str,
    content: str,
    *,
    lock: str = LOCK_NONE,
    locked_at: str | None = None,
    created_at: str = "2026-04-26T00:00:00Z",
    origin: str = "unknown",
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        locked_at=locked_at,
        created_at=created_at,
        last_retrieved_at=None,
        origin=origin,
    )


# ---------------------------------------------------------------------------
# _slot_conflict — spec-matching public signature
# ---------------------------------------------------------------------------


def test_same_slot_same_value_no_flag() -> None:
    """Belief and locked share the same numeric slot value → no conflict."""
    hit = _mk("hit", "alpha = 0.5 prior")
    locked = _mk("locked", "alpha = 0.5 in config", lock=LOCK_USER, locked_at="2026-04-26T00:00:00Z")
    assert _slot_conflict(hit, [locked]) is None


def test_same_slot_different_value_flags() -> None:
    """Numeric slot with values outside tolerance → conflict fires."""
    hit = _mk("hit", "alpha = 0.5 prior")
    locked = _mk("locked", "alpha = 1.0 in config", lock=LOCK_USER, locked_at="2026-04-26T00:00:00Z")
    result = _slot_conflict(hit, [locked])
    assert result == "locked"


def test_enum_same_group_no_flag() -> None:
    """sync and synchronous are same group_id → no conflict."""
    hit = _mk("hit", "use sync mode for hot path")
    locked = _mk("locked", "synchronous everywhere", lock=LOCK_USER, locked_at="2026-04-26T00:00:00Z")
    assert _slot_conflict(hit, [locked]) is None


def test_enum_disjoint_groups_flags() -> None:
    """sync vs async are disjoint groups → conflict fires."""
    hit = _mk("hit", "async execution model")
    locked = _mk("locked", "synchronous on hot path", lock=LOCK_USER, locked_at="2026-04-26T00:00:00Z")
    result = _slot_conflict(hit, [locked])
    assert result == "locked"


def test_different_slots_no_flag() -> None:
    """Belief and locked have no overlapping slots → no conflict."""
    hit = _mk("hit", "synchronous on hot path")
    locked = _mk("locked", "alpha = 0.5 prior", lock=LOCK_USER, locked_at="2026-04-26T00:00:00Z")
    assert _slot_conflict(hit, [locked]) is None


def test_empty_locked_set_no_flag() -> None:
    """Empty locked_set → always None."""
    hit = _mk("hit", "alpha = 0.5 prior")
    assert _slot_conflict(hit, []) is None


def test_multiple_locked_first_match_wins() -> None:
    """When belief conflicts with two locked beliefs, first in sequence wins."""
    hit = _mk("hit", "alpha = 0.5 prior")
    locked_a = _mk("locked_a", "alpha = 1.0 first", lock=LOCK_USER, locked_at="2026-04-26T00:00:00Z")
    locked_b = _mk("locked_b", "alpha = 2.0 second", lock=LOCK_USER, locked_at="2026-04-26T00:00:00Z")
    result = _slot_conflict(hit, [locked_a, locked_b])
    assert result == "locked_a"


def test_no_conflict_with_first_locked_checks_second() -> None:
    """When first locked doesn't conflict but second does, returns second."""
    hit = _mk("hit", "alpha = 0.5 prior")
    # locked_a has no overlapping slots with hit
    locked_a = _mk("locked_a", "synchronous on hot path", lock=LOCK_USER, locked_at="2026-04-26T00:00:00Z")
    locked_b = _mk("locked_b", "alpha = 1.0 in config", lock=LOCK_USER, locked_at="2026-04-26T00:00:00Z")
    result = _slot_conflict(hit, [locked_a, locked_b])
    assert result == "locked_b"


# ---------------------------------------------------------------------------
# _slot_conflict_preextracted — hot-path variant
# ---------------------------------------------------------------------------


def test_preextracted_same_as_spec_function_no_conflict() -> None:
    hit = _mk("hit", "alpha = 0.5 prior")
    locked = _mk("locked", "alpha = 0.5 in config", lock=LOCK_USER, locked_at="2026-04-26T00:00:00Z")
    pairs = [(locked, extract_values(locked.content))]
    assert _slot_conflict_preextracted(hit, pairs) is None


def test_preextracted_same_as_spec_function_with_conflict() -> None:
    hit = _mk("hit", "alpha = 0.5 prior")
    locked = _mk("locked", "alpha = 1.0 in config", lock=LOCK_USER, locked_at="2026-04-26T00:00:00Z")
    pairs = [(locked, extract_values(locked.content))]
    assert _slot_conflict_preextracted(hit, pairs) == "locked"


def test_preextracted_empty_no_flag() -> None:
    hit = _mk("hit", "alpha = 0.5 prior")
    assert _slot_conflict_preextracted(hit, []) is None


# ---------------------------------------------------------------------------
# Latency budget
#
# Spec: ≤2 ms p95 added per query with a small locked set.
# We relax to ≤5 ms here to accommodate slower CI hardware while still
# catching pathological regressions. The 2 ms target is met on developer
# hardware — measured ~0.1 ms p95 in local runs.
# ---------------------------------------------------------------------------

_LATENCY_THRESHOLD_MS = 5.0  # relaxed from spec's 2 ms for CI tolerance


def test_latency_budget() -> None:
    """Median round of _slot_conflict_preextracted over 10 hits × 5 locked ≤ 5 ms.

    The median, because a busy host preempts a minority of rounds: under 44
    busy-loop processes the p95 reached 28 ms while the median stayed under
    0.3 ms, and it stayed there at 88 (#1853). A slowdown of over 5 ms on half or
    more of the rounds moves the median past the budget. A slowdown on fewer
    stays uncaught: 50 ms every 100th call, 10 ms every 50th call and a gc pass
    every 100th call all pass. 10 ms every 20th call lands on exactly half the
    rounds, so it fails only just.
    """
    # Build 5 locked beliefs with distinct numeric slots
    locked_beliefs = [
        _mk(f"locked_{i}", f"param_{i} = {float(i + 1)} baseline",
            lock=LOCK_USER, locked_at="2026-04-26T00:00:00Z")
        for i in range(5)
    ]
    locked_pairs = [(b, extract_values(b.content)) for b in locked_beliefs]

    # Build 10 hit beliefs (no conflicts — tests the non-conflicting path too)
    hits = [
        _mk(f"hit_{i}", f"depth = {float(i + 10)} max")
        for i in range(10)
    ]

    ITERATIONS = 100
    elapsed_ms_list: list[float] = []

    for _ in range(ITERATIONS):
        t0 = time.perf_counter_ns()
        for h in hits:
            _slot_conflict_preextracted(h, locked_pairs)
        t1 = time.perf_counter_ns()
        elapsed_ms_list.append((t1 - t0) / 1_000_000)

    median_ms = statistics.median(elapsed_ms_list)

    assert median_ms <= _LATENCY_THRESHOLD_MS, (
        f"median of {ITERATIONS} rounds took {median_ms:.3f} ms, over the "
        f"{_LATENCY_THRESHOLD_MS} ms threshold (10 hits × 5 locked)"
    )


def _count_hot_path_work(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    """Count the two calls that do the hot path's work, and still make them."""
    counts = {"extract_values": 0, "find_conflicts": 0}
    real_extract = contradiction.extract_values
    real_find = contradiction.find_conflicts

    def counted_extract(text: str) -> ValueSlots:
        counts["extract_values"] += 1
        return real_extract(text)

    def counted_find(a: ValueSlots, b: ValueSlots) -> list[SlotConflict]:
        counts["find_conflicts"] += 1
        return real_find(a, b)

    monkeypatch.setattr(contradiction, "extract_values", counted_extract)
    monkeypatch.setattr(contradiction, "find_conflicts", counted_find)
    return counts


def test_work_per_call_without_a_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hit extracts at most once and compares at most once per locked belief (#1853).

    A count holds on any host. These are upper bounds, so a refactor that does
    less passes and one that re-extracts the locked set fails.
    """
    pairs = [
        (_mk(f"locked_{i}", f"param_{i} = {float(i + 1)} baseline"),
         extract_values(f"param_{i} = {float(i + 1)} baseline"))
        for i in range(5)
    ]
    hits = [_mk(f"hit_{i}", f"depth = {float(i + 10)} max") for i in range(10)]
    counts = _count_hot_path_work(monkeypatch)

    results = [_slot_conflict_preextracted(h, pairs) for h in hits]

    assert results == [None] * 10
    assert counts["extract_values"] <= 10
    assert counts["find_conflicts"] <= 50


def test_work_per_call_stops_at_the_first_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conflict on the first locked belief ends that hit's comparisons (#1853)."""
    pairs = [
        (_mk(f"locked_{i}", f"depth = {float(i + 100)} max"),
         extract_values(f"depth = {float(i + 100)} max"))
        for i in range(5)
    ]
    hits = [_mk(f"hit_{i}", f"depth = {float(i + 10)} max") for i in range(10)]
    counts = _count_hot_path_work(monkeypatch)

    results = [_slot_conflict_preextracted(h, pairs) for h in hits]

    assert results == ["locked_0"] * 10
    assert counts["extract_values"] <= 10
    assert counts["find_conflicts"] <= 10
