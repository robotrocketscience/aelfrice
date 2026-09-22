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


def _run_at_weight(fixtures: list[dict], weight: str | None):
    """Run the harness with `AELFRICE_POSTERIOR_WEIGHT` set, or unset.

    The weight goes through the environment rather than a kwarg so the
    shipped resolver runs, which is the path the gate is claiming to
    measure.
    """
    env = pytest.MonkeyPatch()
    if weight is not None:
        env.setenv("AELFRICE_POSTERIOR_WEIGHT", weight)
    try:
        return run_calibration_on_fixtures(fixtures)
    finally:
        env.undo()


def test_disabling_the_blend_changes_the_ranking(
    fixtures: list[dict],
) -> None:
    """The gate's whole premise, asserted directly rather than in CI.

    Compares **rankings**, not aggregate metrics (#1584). The metrics are
    multiset statistics of the pooled observations, so they agree across
    two runs whose per-query rank moves cancel — on this corpus that is
    exactly what `posterior_weight` 0.0 and 1.5 do. Asserting on them
    tests the weaker claim "the blend moves this corpus's aggregates",
    and reports any failure as inertness, which is a different thing.
    """
    on = _run_at_weight(fixtures, None)
    off = _run_at_weight(fixtures, "0.0")

    assert on.rankings and off.rankings, "the harness recorded no rankings"
    assert on.rankings != off.rankings, (
        "the shipped posterior_weight retrieves the same ranking as "
        "posterior_weight=0.0 on every query, so the blend is inert on "
        "this corpus and the calibration gate measures nothing. "
        f"rankings={on.rankings!r}"
    )


def test_equal_metrics_do_not_imply_an_unchanged_ranking(
    fixtures: list[dict],
) -> None:
    """Why the guard above cannot be written against the metrics (#1584).

    `posterior_weight` 1.5 lands in a band whose aggregates are
    byte-identical to the 0.0 arm's, because the relevant belief falls
    1→2 on one query and rises 2→1 on another and the two cancel in the
    pool. A third query reorders two non-relevant candidates, which the
    labels cannot register at all.

    This is an aggregation collision, not inertness. If the shipped
    default were ever moved into that band, a metric-based guard would
    fail with a message naming the wrong cause — so this test pins the
    collision itself, and fails if the corpus stops exhibiting it and
    the weaker guard silently becomes sufficient again.
    """
    off = _run_at_weight(fixtures, "0.0")
    collided = _run_at_weight(fixtures, "1.5")

    assert (collided.roc_auc, collided.spearman_rho) == (
        off.roc_auc,
        off.spearman_rho,
    ), (
        "posterior_weight=1.5 no longer collides with the 0.0 arm's "
        "aggregates. The corpus changed; re-derive the band edges before "
        "trusting any figure in #1584."
    )
    assert collided.rankings != off.rankings, (
        "posterior_weight=1.5 now retrieves the 0.0 ranking as well as "
        "its metrics, so the collision this test documents is gone and "
        "the blend really is inert at 1.5"
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
