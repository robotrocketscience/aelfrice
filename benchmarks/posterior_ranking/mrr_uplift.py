"""MRR uplift evaluator for the posterior-ranking eval harness.

Implements the 10-round MRR uplift contract from
docs/design/v2_posterior_ranking_residual.md § Slice 1.

Round 0: baseline retrieve (no feedback applied).  Record mrr_0 = 1/rank
of the known_belief_content in the top-K results (0 if not found).

Rounds 1..10: apply one synthetic positive feedback event per query against
the top-1 result if it matches the known item, against the known item itself
if not.  Re-retrieve.  Record mrr_i.

Multi-seed runner reports (mean, ±2σ) over n_seeds independent seeds.

Pass criterion:
  mrr_uplift = mrr_10 - mrr_0 >= threshold
  AND no round shows regression below mrr_0 - 0.01
"""
from __future__ import annotations

import math
import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from aelfrice.feedback import apply_feedback
from aelfrice.graph_spectral import GraphEigenbasisCache
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

# How many retrieve-then-feedback rounds to run after round 0.
N_ROUNDS: int = 10
# Retrieve up to this many beliefs per call when computing rank.
DEFAULT_TOP_K: int = 20
# Default pass threshold (spec: +0.05).
DEFAULT_MRR_THRESHOLD: float = 0.05
# Tolerance for regression detection (spec: mrr_0 - 0.01).
REGRESSION_FLOOR_DELTA: float = 0.01


@dataclass
class MRRUpliftResult:
    """Per-fixture-set MRR uplift result for one seed."""

    mrr_0: float
    mrr_per_round: list[float]
    mrr_uplift: float
    seed: int
    pass_threshold: float
    passed: bool

    @property
    def mrr_10(self) -> float:
        """MRR after the final round."""
        return self.mrr_per_round[-1] if self.mrr_per_round else self.mrr_0


@dataclass
class MultiSeedReport:
    """Aggregated MRR uplift over multiple seeds."""

    results: list[MRRUpliftResult]
    mean_uplift: float
    std_uplift: float
    # ±2σ band
    uplift_lo: float
    uplift_hi: float
    pass_threshold: float
    passed: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_belief(bid: str, content: str) -> Belief:
    """Construct a minimal Belief suitable for insertion."""
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=0.5,
        beta=0.5,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
    )


def _mrr_for_belief(
    store: MemoryStore,
    query: str,
    known_content: str,
    top_k: int = DEFAULT_TOP_K,
    *,
    eigenbasis_cache: GraphEigenbasisCache | None = None,
    heat_kernel: bool = False,
) -> float:
    """Return 1/rank of the known belief in the retrieve() results (0 if absent)."""
    results: list[Belief] = retrieve(
        store,
        query,
        l1_limit=top_k,
        entity_index_enabled=False,
        bfs_enabled=False,
        posterior_weight=None,  # use default 0.5
        heat_kernel_enabled=heat_kernel,
        eigenbasis_cache=eigenbasis_cache,
    )
    for rank, belief in enumerate(results, start=1):
        if belief.content == known_content:
            return 1.0 / rank
    return 0.0


def _build_store(
    fixture: dict[str, object],
    seed: int,
    *,
    noise_shuffle: bool = True,
) -> tuple[MemoryStore, str]:
    """Build an in-memory store from a fixture entry.

    Returns (store, known_belief_id).
    """
    rng = random.Random(seed)
    store = MemoryStore(":memory:")
    known_content: str = str(fixture["known_belief_content"])
    noise_contents: list[str] = list(fixture["noise_belief_contents"])  # type: ignore[arg-type]

    fid: str = str(fixture["id"])
    known_id: str = f"{fid}_known"
    store.insert_belief(_make_belief(known_id, known_content))

    if noise_shuffle:
        rng.shuffle(noise_contents)
    for i, nc in enumerate(noise_contents):
        store.insert_belief(_make_belief(f"{fid}_noise_{i}", nc))

    return store, known_id


def run_single_seed(
    fixtures: list[dict[str, object]],
    seed: int,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_MRR_THRESHOLD,
    *,
    heat_kernel: bool = False,
) -> MRRUpliftResult:
    """Run the 10-round MRR uplift evaluator for one seed.

    Builds one in-memory store per fixture, runs retrieval rounds, applies
    synthetic feedback, aggregates MRR across all fixtures per round.

    Synthetic feedback contract:
      - If the top-1 result matches the known item: positive feedback on top-1.
      - Else: positive feedback on the known item directly.
    This simulates a user approving the right answer.
    """
    n_fixtures = len(fixtures)
    if n_fixtures == 0:
        raise ValueError("fixtures must not be empty")

    # Per-seed throwaway temp dir for the eigenbasis .npz files when
    # heat_kernel is on. Cache files are rebuilt after every feedback
    # round (store mutation invalidates them); the cleanup is handled
    # by the TemporaryDirectory at function exit.
    _tmp_dir = tempfile.TemporaryDirectory(prefix="aelf_pr_eb_")
    _tmp_path = Path(_tmp_dir.name)

    # Build stores once; share across rounds.
    stores_ids: list[tuple[MemoryStore, str, dict[str, object], GraphEigenbasisCache | None]] = []
    for idx, fx in enumerate(fixtures):
        store, known_id = _build_store(fx, seed)
        cache: GraphEigenbasisCache | None = None
        if heat_kernel:
            cache = GraphEigenbasisCache(
                store=store, path=_tmp_path / f"eb_{seed}_{idx}.npz",
            )
            cache.build()
        stores_ids.append((store, known_id, fx, cache))

    def _refresh_caches() -> None:
        if not heat_kernel:
            return
        for st, _kid, _fx, cache in stores_ids:
            if cache is not None and cache.is_stale():
                cache.build()

    def _mean_mrr(round_idx: int) -> float:
        """Compute mean MRR across all fixtures for the current store state."""
        del round_idx  # round tracked externally; store state is what matters
        _refresh_caches()
        total = 0.0
        for st, _kid, fx, cache in stores_ids:
            total += _mrr_for_belief(
                st, str(fx["query"]), str(fx["known_belief_content"]), top_k,
                eigenbasis_cache=cache, heat_kernel=heat_kernel,
            )
        return total / n_fixtures

    mrr_0 = _mean_mrr(0)
    regression_floor = mrr_0 - REGRESSION_FLOOR_DELTA

    mrr_per_round: list[float] = []
    any_regression = False

    for _rnd in range(N_ROUNDS):
        _refresh_caches()
        # Apply synthetic feedback to each fixture's store.
        for st, known_id, fx, cache in stores_ids:
            query = str(fx["query"])
            known_content = str(fx["known_belief_content"])
            results: list[Belief] = retrieve(
                st,
                query,
                l1_limit=top_k,
                entity_index_enabled=False,
                bfs_enabled=False,
                posterior_weight=None,
                heat_kernel_enabled=heat_kernel,
                eigenbasis_cache=cache,
            )
            top1 = results[0] if results else None
            if top1 is not None and top1.content == known_content:
                target_id = top1.id
            else:
                target_id = known_id
            apply_feedback(
                st,
                target_id,
                valence=1.0,
                source="eval_synthetic",
            )

        round_mrr = _mean_mrr(_rnd + 1)
        mrr_per_round.append(round_mrr)
        if round_mrr < regression_floor:
            any_regression = True

    mrr_10 = mrr_per_round[-1]
    uplift = mrr_10 - mrr_0
    passed = (uplift >= threshold) and (not any_regression)

    # Close stores.
    for st, _, _, _ in stores_ids:
        st.close()
    _tmp_dir.cleanup()

    return MRRUpliftResult(
        mrr_0=mrr_0,
        mrr_per_round=mrr_per_round,
        mrr_uplift=uplift,
        seed=seed,
        pass_threshold=threshold,
        passed=passed,
    )


def run_multi_seed(
    fixtures: list[dict[str, object]],
    n_seeds: int = 5,
    threshold: float = DEFAULT_MRR_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    base_seed: int = 0,
    *,
    heat_kernel: bool = False,
) -> MultiSeedReport:
    """Run the uplift evaluator across n_seeds seeds.

    Seeds are derived deterministically from base_seed.
    Returns aggregated (mean, ±2σ) uplift report.
    """
    results: list[MRRUpliftResult] = []
    for i in range(n_seeds):
        seed = base_seed + i
        r = run_single_seed(
            fixtures, seed=seed, top_k=top_k, threshold=threshold,
            heat_kernel=heat_kernel,
        )
        results.append(r)

    uplifts = [r.mrr_uplift for r in results]
    mean_uplift = sum(uplifts) / len(uplifts)

    if len(uplifts) > 1:
        variance = sum((u - mean_uplift) ** 2 for u in uplifts) / (len(uplifts) - 1)
        std_uplift = math.sqrt(variance)
    else:
        std_uplift = 0.0

    uplift_lo = mean_uplift - 2.0 * std_uplift
    uplift_hi = mean_uplift + 2.0 * std_uplift

    # Overall pass: mean uplift meets threshold AND all seeds pass.
    passed = (mean_uplift >= threshold) and all(r.passed for r in results)

    return MultiSeedReport(
        results=results,
        mean_uplift=mean_uplift,
        std_uplift=std_uplift,
        uplift_lo=uplift_lo,
        uplift_hi=uplift_hi,
        pass_threshold=threshold,
        passed=passed,
    )


def load_fixtures(path: Path | str) -> list[dict[str, object]]:
    """Load fixtures from a JSON Lines file."""
    import json

    p = Path(path)
    fixtures: list[dict[str, object]] = []
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                fixtures.append(json.loads(line))
    return fixtures
