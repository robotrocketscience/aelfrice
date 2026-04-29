"""Posterior-ranking eval runner: wires MRR uplift + ECE against a fixture set.

Entry point used by both tests and the `aelf bench posterior-residual` CLI.

Usage:
    run(fixtures_path, n_seeds=5) -> dict with keys "mrr", "ece", "overall_pass"

The synthetic feedback stream is defined as follows:
  For each (query, belief, rank) triple produced by round-0 retrieval, the
  belief is marked as "received_positive" if it is the known_belief_content
  for its fixture.  This gives the ECE scorer a clean signal: the known
  item always gets positive feedback; noise items do not.  The posterior
  parameters (alpha, beta) used for ECE are read AFTER all feedback rounds
  complete, reflecting the accumulated posterior shift.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.posterior_ranking.ece import (
    DEFAULT_ECE_THRESHOLD,
    ECEResult,
    compute_ece_from_stores,
)
from benchmarks.posterior_ranking.mrr_uplift import (
    DEFAULT_MRR_THRESHOLD,
    DEFAULT_TOP_K,
    N_ROUNDS,
    MultiSeedReport,
    _build_store,
    _mrr_for_belief,
    load_fixtures,
    run_multi_seed,
)
from aelfrice.feedback import apply_feedback
from aelfrice.models import Belief
from aelfrice.retrieval import retrieve


def _build_ece_observations(
    fixtures: list[dict[str, object]],
    seed: int,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, object]]:
    """Replay the synthetic feedback stream and collect ECE observations.

    For each round, for each belief in the retrieved top-K, record a
    (alpha, beta, received_positive) triple where:
      - alpha, beta: the belief's posterior parameters AT RETRIEVAL TIME
        (before feedback is applied in this round).
      - received_positive: 1.0 if this belief received positive feedback in
        this round, 0.0 otherwise.

    This mirrors the real-feedback ECE contract: predicted probability at
    query time vs empirical positive-feedback outcome.  The per-round
    sampling captures how well-calibrated the posterior_mean is as a
    predictor of user approval across ALL rounds, not just one snapshot.
    """
    observations: list[dict[str, object]] = []

    for fx in fixtures:
        store, known_id = _build_store(fx, seed)
        query = str(fx["query"])
        known_content = str(fx["known_belief_content"])

        # Round 0 is baseline (no feedback yet); include it.
        for _rnd in range(N_ROUNDS + 1):
            results: list[Belief] = retrieve(
                store,
                query,
                l1_limit=top_k,
                entity_index_enabled=False,
                bfs_enabled=False,
                posterior_weight=None,
            )

            top1 = results[0] if results else None
            if top1 is not None and top1.content == known_content:
                feedback_target_id = top1.id
            else:
                feedback_target_id = known_id

            # Record observation for each retrieved belief BEFORE feedback.
            for b in results:
                received_positive = (b.id == feedback_target_id)
                observations.append({
                    "alpha": b.alpha,
                    "beta": b.beta,
                    "received_positive": received_positive,
                })

            # Apply feedback (skip on the last pass so we don't overshoot
            # the N_ROUNDS contract used by the MRR scorer).
            if _rnd < N_ROUNDS:
                apply_feedback(
                    store,
                    feedback_target_id,
                    valence=1.0,
                    source="eval_synthetic",
                    propagate=False,
                )

        store.close()

    return observations


def run(
    fixtures_path: Path | str,
    n_seeds: int = 5,
    mrr_threshold: float = DEFAULT_MRR_THRESHOLD,
    ece_threshold: float = DEFAULT_ECE_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    base_seed: int = 0,
) -> dict[str, Any]:
    """Run both MRR uplift and ECE scorers against a fixture file.

    Returns:
        {
            "mrr": MultiSeedReport (as dataclass),
            "ece": ECEResult (as dataclass),
            "overall_pass": bool,
        }

    ``overall_pass`` is True when both mrr.passed and ece.passed.

    The fixtures_path must be a JSON Lines file; each line is one fixture dict
    with keys: id, query, known_belief_content, noise_belief_contents.
    """
    fixtures = load_fixtures(fixtures_path)

    mrr_report: MultiSeedReport = run_multi_seed(
        fixtures,
        n_seeds=n_seeds,
        threshold=mrr_threshold,
        top_k=top_k,
        base_seed=base_seed,
    )

    # Collect ECE observations using seed 0 (deterministic reference).
    observations = _build_ece_observations(fixtures, seed=base_seed, top_k=top_k)
    ece_result: ECEResult = compute_ece_from_stores(observations, threshold=ece_threshold)

    overall_pass = mrr_report.passed and ece_result.passed

    return {
        "mrr": mrr_report,
        "ece": ece_result,
        "overall_pass": overall_pass,
    }


def run_as_dict(
    fixtures_path: Path | str,
    n_seeds: int = 5,
    mrr_threshold: float = DEFAULT_MRR_THRESHOLD,
    ece_threshold: float = DEFAULT_ECE_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    base_seed: int = 0,
) -> dict[str, Any]:
    """Same as run() but serializes dataclasses to plain dicts for JSON output."""
    result = run(
        fixtures_path,
        n_seeds=n_seeds,
        mrr_threshold=mrr_threshold,
        ece_threshold=ece_threshold,
        top_k=top_k,
        base_seed=base_seed,
    )
    return {
        "mrr": asdict(result["mrr"]),
        "ece": asdict(result["ece"]),
        "overall_pass": result["overall_pass"],
    }
