"""The calibration corpus must be able to see the ranker (#1160).

Before this, `build_calibration_store` built every belief at
`alpha=beta=0.5`. The scoring term is
`posterior_weight * log(posterior_mean)`, so a constant posterior is a
constant *offset* across candidates and cannot reorder anything:
`AELFRICE_POSTERIOR_WEIGHT` at 0.0, 1.0 and 5.0 emitted byte-identical
metrics, and the one byte-exact ranking baseline in CI was provably
blind to the Bayesian rerank it is named for.

These tests pin the property that makes the gate work, so a corpus edit
that flattens the posteriors fails here rather than silently restoring
the blindness.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.eval_harness import (
    DEFAULT_CALIBRATION_CORPUS,
    build_calibration_store,
    load_calibration_fixtures,
    run_calibration_on_fixtures,
)
from aelfrice.scoring import posterior_mean

_BASE = Path(__file__).resolve().parents[1] / "benchmarks" / "posterior_ranking"
_ON = _BASE / "baseline.json"
_OFF = _BASE / "baseline_posterior_off.json"


@pytest.fixture(scope="module")
def fixtures() -> list[dict]:
    return load_calibration_fixtures(DEFAULT_CALIBRATION_CORPUS)


def test_corpus_posteriors_are_varied(fixtures: list[dict]) -> None:
    """A constant posterior is a constant offset — it cannot reorder."""
    for row in fixtures:
        means = [posterior_mean(*row["known_posterior"])] + [
            posterior_mean(*ab) for ab in row["noise_posteriors"]
        ]
        assert len(set(means)) > 1, (
            f"{row['id']}: every candidate shares a posterior, so the "
            f"blend term is a constant offset for this query"
        )


def test_the_relevant_belief_is_not_always_the_best_posterior(
    fixtures: list[dict],
) -> None:
    """Otherwise raising the weight monotonically improves every metric.

    A corpus where the known belief always carries the top posterior
    would make the gate reward cranking `posterior_weight` rather than
    detecting that the blend went inert.
    """
    ranks = []
    for row in fixtures:
        known = posterior_mean(*row["known_posterior"])
        noise = [posterior_mean(*ab) for ab in row["noise_posteriors"]]
        ranks.append(sorted([known] + noise, reverse=True).index(known) + 1)

    assert max(ranks) > 1, "the known belief is top-posterior everywhere"
    assert min(ranks) == 1, "the known belief is never top-posterior"


def test_posteriors_reach_the_store(fixtures: list[dict]) -> None:
    """The fixture fields must survive `build_calibration_store`.

    Noise contents are shuffled, so a posterior attached by position
    rather than carried with its content would land on the wrong belief.
    """
    row = next(r for r in fixtures if r["id"] == "q2")
    store = build_calibration_store(row, seed=0)
    fid = row["id"]

    ids = [f"{fid}_known"] + [
        f"{fid}_noise_{i}" for i in range(len(row["noise_belief_contents"]))
    ]
    by_content: dict[str, tuple[float, float]] = {}
    for bid in ids:
        belief = store.get_belief(bid)
        assert belief is not None, f"{bid} was not inserted"
        by_content[belief.content] = (belief.alpha, belief.beta)
    assert len(by_content) == len(ids), "contents collided"

    assert by_content[row["known_belief_content"]] == tuple(
        row["known_posterior"]
    )
    for content, ab in zip(
        row["noise_belief_contents"], row["noise_posteriors"]
    ):
        assert by_content[content] == tuple(ab), (
            f"{content[:40]!r} carries {by_content[content]}, expected "
            f"{tuple(ab)} — the shuffle decoupled content from posterior"
        )


def test_disabling_the_blend_changes_the_metrics(
    fixtures: list[dict],
) -> None:
    """The gate's whole premise, asserted directly rather than in CI."""
    on = run_calibration_on_fixtures(fixtures)
    off_env = pytest.MonkeyPatch()
    off_env.setenv("AELFRICE_POSTERIOR_WEIGHT", "0.0")
    try:
        off = run_calibration_on_fixtures(fixtures)
    finally:
        off_env.undo()

    assert (on.roc_auc, on.spearman_rho) != (off.roc_auc, off.spearman_rho), (
        "posterior_weight=0.0 produced identical metrics — the rerank is "
        "inert on this corpus and the calibration gate measures nothing"
    )


def test_the_two_pinned_baselines_differ() -> None:
    """Equal baselines would mean the committed gate is inert.

    Catches the case where someone flattens the corpus and regenerates
    both files together, which would otherwise look self-consistent.
    """
    on = json.loads(_ON.read_text(encoding="utf-8"))
    off = json.loads(_OFF.read_text(encoding="utf-8"))
    assert on != off
    assert on["roc_auc"] != off["roc_auc"]
