"""#1143 clock seam: a pinned `now_ts` reads no wall clock in the
tiered retrieval path.

Contract under test: `retrieve_v2(now_ts=...)` threads the pin through
`retrieve_with_tiers` and `_l1_hits`, so the γ resolver, the expansion
gate, and the L1 meta-resolver / signal write all see the same frozen
timestamp — no site re-reads `time.time()`. Pinned calls are also
rank-reproducible, and the two function bodies carry no direct
wall-clock reads (the source assertion covers flag-gated sites the
runtime tests can't reach in a default config).
"""
from __future__ import annotations

import inspect
import time as _real_time
from pathlib import Path
from types import SimpleNamespace

import pytest

from aelfrice import retrieval
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

PINNED_TS = 1_700_000_000


def _mk_belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash="h_" + bid,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-07-21T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(store: MemoryStore) -> None:
    store.insert_belief(_mk_belief("b1", "the quick brown fox jumps"))
    store.insert_belief(_mk_belief("b2", "sqlite stores beliefs durably"))
    store.insert_belief(_mk_belief("b3", "the fox likes sqlite"))


def _no_wall_clock_time() -> SimpleNamespace:
    """A `time` module stand-in whose `time()` raises.

    `perf_counter` passes through — it measures latency, not the
    calendar clock the seam pins.
    """
    def _boom() -> float:
        raise AssertionError(
            "wall clock read despite pinned now_ts (#1143 seam leak)"
        )
    return SimpleNamespace(
        time=_boom,
        perf_counter=_real_time.perf_counter,
    )


def test_pinned_tiers_call_reads_no_wall_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _seed(store)
        monkeypatch.setattr(retrieval, "time", _no_wall_clock_time())
        out, locked, l25, l1, chains = retrieval.retrieve_with_tiers(
            store, "fox sqlite", now_ts=PINNED_TS,
        )
        assert [b.id for b in out]
    finally:
        store.close()


def test_pinned_retrieve_v2_reads_no_wall_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _seed(store)
        monkeypatch.setattr(retrieval, "time", _no_wall_clock_time())
        result = retrieval.retrieve_v2(
            store, "fox sqlite", now_ts=PINNED_TS,
        )
        assert [b.id for b in result.beliefs]
    finally:
        store.close()


def test_unpinned_call_still_works(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _seed(store)
        out, *_ = retrieval.retrieve_with_tiers(store, "fox sqlite")
        assert [b.id for b in out]
    finally:
        store.close()


def test_pinned_ranking_is_reproducible(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        _seed(store)
        runs = [
            [
                b.id
                for b in retrieval.retrieve_with_tiers(
                    store, "fox sqlite", now_ts=PINNED_TS,
                )[0]
            ]
            for _ in range(3)
        ]
        assert runs[0] == runs[1] == runs[2]
    finally:
        store.close()


def test_function_bodies_carry_no_direct_clock_reads() -> None:
    """Flag-gated sites (the γ resolver arm) never fire in a default
    config, so pin the acceptance criterion at the source level: the
    only `time.time()` in either body is the single seam fallback.
    """
    for fn in (retrieval._l1_hits, retrieval.retrieve_with_tiers):
        src = inspect.getsource(fn)
        assert src.count("time.time()") == 1, fn.__name__
        assert "if now_ts is not None else int(time.time())" in src, (
            fn.__name__
        )
