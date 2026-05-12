"""Bench-gated HRR cold-start timing gates (#697, sub-task of #553).

Both tests are gated by the ``bench_gated`` autouse marker and skip when
``AELFRICE_CORPUS_ROOT`` is unset (normal CI).  Run the gates explicitly
by setting the env var to any existing directory.

Acceptance criteria (docs/feature-hrr-integration.md §"Acceptance criteria"):
  - Warm cold-start (persist-on): ≤ 1.0 s at N=50k
  - Rebuild cold-start (persist-off): ≤ 38.0 s at N=50k
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.bench_gate._hrr_synthetic_store import build_n50k_store

_SEED = 42
_N = 50_000
_DIM = 512  # DEFAULT_DIM post-#538; matches production default

# Timing ceilings (seconds) per spec.
_WARM_CEILING_S = 1.0
_REBUILD_CEILING_S = 38.0


@pytest.mark.bench_gated
@pytest.mark.timeout(120)  # build phase takes ~38s; warm load ≤ 1s; 120s total headroom
def test_hrr_cold_start_warm_load_under_one_second_at_50k(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm cold-start (mmap load) completes in ≤ 1 s at N=50k.

    Protocol:
    1. Build synthetic N=50k store, prime persist dir via first cache.get().
    2. Confirm struct.npy written to <tmp_path>/.hrr_struct_index/struct.npy.
    3. Drop references to first cache and store (no warm in-process state).
    4. Open a fresh MemoryStore + HRRStructIndexCache pointing at the same db.
    5. Time the first cache.get() — must use the mmap load path, not rebuild.
    6. Assert wall-clock ≤ 1.0 s.
    7. Call probe() once to confirm the loaded index is usable (exercises the
       mmap-backed struct matrix; not timed — correctness, not perf).
    """
    from aelfrice.hrr_index import HRRStructIndexCache
    from aelfrice.store import MemoryStore

    # Ensure persistence is enabled (default; guard against env leakage).
    monkeypatch.delenv("AELFRICE_HRR_PERSIST", raising=False)

    memory_db = tmp_path / "memory.db"
    store_path = str(memory_db)

    # --- Phase 1: prime the persist dir ---
    store1 = build_n50k_store(memory_db, n_beliefs=_N, seed=_SEED)
    cache1 = HRRStructIndexCache(
        store=store1, dim=_DIM, store_path=store_path, seed=_SEED
    )
    cache1.get()  # builds and saves struct.npy

    persist_dir = tmp_path / ".hrr_struct_index"
    assert (persist_dir / "struct.npy").is_file(), (
        "persist dir not written — warm-load test cannot proceed"
    )

    store1.close()
    del cache1  # drop all references; next get() must go through load, not cache

    # --- Phase 2: fresh process simulation (fresh store + cache) ---
    store2 = MemoryStore(store_path)
    cache2 = HRRStructIndexCache(
        store=store2, dim=_DIM, store_path=store_path, seed=_SEED
    )

    t0 = time.perf_counter()
    idx = cache2.get()
    elapsed = time.perf_counter() - t0

    # Correctness smoke: probe must return results (mmap lazy-load validation).
    # "b000050" has a CONTRADICTS edge from "b000000" (offset 50 in the helper).
    hits = idx.probe("CONTRADICTS", "b000050", top_k=5)
    assert hits, (
        f"probe returned no hits on mmap-loaded index (N={_N}, elapsed={elapsed:.3f}s)"
    )

    store2.close()

    assert elapsed <= _WARM_CEILING_S, (
        f"HRR warm cold-start took {elapsed:.3f}s, exceeds {_WARM_CEILING_S}s ceiling "
        f"at N={_N}. mmap load path may have regressed (check for accidental rebuild)."
    )


@pytest.mark.bench_gated
@pytest.mark.timeout(120)  # rebuild ceiling is 38s; 120s gives 3× headroom on slow CI
def test_hrr_cold_start_rebuild_under_38s_at_50k_persist_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rebuild cold-start (AELFRICE_HRR_PERSIST=0) completes in ≤ 38 s at N=50k.

    This is the negative-control rail: it locks in the rebuild path's ceiling
    so a regression in HRRStructIndex.build() shows up here even when no
    warm-load is available.

    No struct.npy should be written — persistence is disabled via env var.
    """
    from aelfrice.hrr_index import HRRStructIndexCache

    monkeypatch.setenv("AELFRICE_HRR_PERSIST", "0")

    memory_db = tmp_path / "memory.db"
    store_path = str(memory_db)

    store = build_n50k_store(memory_db, n_beliefs=_N, seed=_SEED)
    cache = HRRStructIndexCache(
        store=store, dim=_DIM, store_path=store_path, seed=_SEED
    )

    t0 = time.perf_counter()
    cache.get()
    elapsed = time.perf_counter() - t0

    persist_dir = tmp_path / ".hrr_struct_index"
    assert not (persist_dir / "struct.npy").exists(), (
        "struct.npy written despite AELFRICE_HRR_PERSIST=0 — opt-out codepath broken"
    )

    store.close()

    assert elapsed <= _REBUILD_CEILING_S, (
        f"HRR rebuild cold-start took {elapsed:.3f}s, exceeds {_REBUILD_CEILING_S}s "
        f"ceiling at N={_N}. HRRStructIndex.build() may have regressed."
    )
