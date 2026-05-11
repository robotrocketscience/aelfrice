"""Unit tests for the bake-off evaluator (#228)."""
from __future__ import annotations

import random

from aelfrice.wonder.evaluator import (
    H0_NULL_RATE,
    H0_PLUS_PP_FLOOR,
    JACCARD_COMPLEMENT,
    JACCARD_REDUNDANCY,
    StrategyMetrics,
    adoption_verdict,
    evaluate_strategy,
    pairwise_jaccard,
)
from aelfrice.wonder.models import (
    STRATEGY_RW,
    STRATEGY_STS,
    STRATEGY_TC,
    Phantom,
)
from aelfrice.wonder.simulator import build_corpus


def _phantom(comp: tuple[str, ...], strategy: str = STRATEGY_RW,
             cost: float = 2.0) -> Phantom:
    return Phantom(
        composition=tuple(sorted(comp)),
        strategy=strategy,
        construction_cost=cost,
    )


def test_evaluate_empty_strategy() -> None:
    corpus = build_corpus(rng=random.Random(0), n_topics=3, n_atoms_per_topic=5)
    m = evaluate_strategy([], corpus)
    assert m.n_phantoms == 0
    assert m.confirmation_rate == 0.0
    assert m.junk_rate == 0.0


def test_evaluate_all_confirms_promotes() -> None:
    corpus = build_corpus(rng=random.Random(0), n_topics=3, n_atoms_per_topic=5)
    same_topic_ids = [a.belief_id for a in corpus.atoms if a.topic == 0]
    p = _phantom((same_topic_ids[0], same_topic_ids[1]))
    # Budget 16 confirms → α=17, well past the α≥12 gate.
    m = evaluate_strategy([p], corpus, feedback_budget_per_phantom=16)
    assert m.confirmation_rate == 1.0
    assert m.junk_rate == 0.0


def test_evaluate_all_junks_marks_junk_rate() -> None:
    # Cross-topic + matching session + matching source_type forces the
    # broadened verdict to junk on all three rules. The original test
    # relied on a build_corpus draw where the two atoms happened to share
    # session and source_type; constructing the corpus explicitly here
    # makes the contract independent of rng state. (#547 B1 broadening.)
    from aelfrice.wonder.simulator import CorpusAtom, SyntheticCorpus
    corpus = SyntheticCorpus(
        atoms=(
            CorpusAtom("a0", 0, "sess_00", "commit_ingest"),
            CorpusAtom("a1", 1, "sess_00", "commit_ingest"),
        )
    )
    p = _phantom(("a0", "a1"))
    m = evaluate_strategy([p], corpus, feedback_budget_per_phantom=16)
    assert m.confirmation_rate == 0.0
    assert m.junk_rate == 1.0


def test_evaluate_retrieval_per_cost_uses_composition_size() -> None:
    corpus = build_corpus(rng=random.Random(0), n_topics=3, n_atoms_per_topic=5)
    same_topic = [a.belief_id for a in corpus.atoms if a.topic == 0]
    # composition size 2, cost 2 → retrieval_per_cost = 1.0
    p = _phantom((same_topic[0], same_topic[1]), cost=2.0)
    m = evaluate_strategy([p], corpus)
    assert m.retrieval_freq_per_cost == 1.0


def test_jaccard_identical_sets_is_one() -> None:
    a = [_phantom(("x", "y")), _phantom(("x", "z"))]
    b = [_phantom(("x", "y")), _phantom(("x", "z"))]
    assert pairwise_jaccard(a, b) == 1.0


def test_jaccard_disjoint_sets_is_zero() -> None:
    a = [_phantom(("x", "y"))]
    b = [_phantom(("u", "v"))]
    assert pairwise_jaccard(a, b) == 0.0


def test_jaccard_partial_overlap() -> None:
    a = [_phantom(("x", "y")), _phantom(("x", "z"))]
    b = [_phantom(("x", "y")), _phantom(("u", "v"))]
    # |A ∩ B| = 1, |A ∪ B| = 3
    assert pairwise_jaccard(a, b) == 1 / 3


def test_jaccard_two_empty_sets_is_one() -> None:
    assert pairwise_jaccard([], []) == 1.0


def _metric(name: str, rate: float, junk: float = 0.0,
            r_per_c: float = 1.0) -> StrategyMetrics:
    return StrategyMetrics(
        strategy=name,
        n_phantoms=10,
        confirmation_rate=rate,
        retrieval_freq_per_cost=r_per_c,
        junk_rate=junk,
        mean_construction_cost=2.0,
    )


def test_verdict_drop_when_all_below_h0() -> None:
    metrics = (
        _metric("RW", 0.01),
        _metric("TC", 0.02),
        _metric("STS", 0.03),
    )
    assert adoption_verdict(metrics, jaccard={}) == "drop"


def test_verdict_defer_on_high_junk_rate() -> None:
    metrics = (
        _metric("RW", 0.5, junk=0.7),
        _metric("TC", 0.2, junk=0.1),
        _metric("STS", 0.3, junk=0.1),
    )
    assert adoption_verdict(metrics, jaccard={}) == "defer"


def test_verdict_defer_when_none_clear_floor() -> None:
    floor = H0_NULL_RATE + H0_PLUS_PP_FLOOR
    metrics = (
        _metric("RW", floor - 0.01),
        _metric("TC", floor - 0.02),
        _metric("STS", H0_NULL_RATE + 0.001),
    )
    assert adoption_verdict(metrics, jaccard={}) == "defer"


def test_verdict_ensemble_when_top_two_complementary() -> None:
    metrics = (
        _metric("RW", 0.3),
        _metric("TC", 0.4),
        _metric("STS", 0.05),
    )
    j = {
        ("RW", "TC"): JACCARD_COMPLEMENT - 0.1,
        ("RW", "STS"): 0.1,
        ("STS", "TC"): 0.1,
    }
    assert adoption_verdict(metrics, j) == "ensemble"


def test_verdict_single_when_top_two_redundant() -> None:
    metrics = (
        _metric("RW", 0.3),
        _metric("TC", 0.4),
        _metric("STS", 0.05),
    )
    j = {
        ("RW", "TC"): JACCARD_REDUNDANCY + 0.1,
        ("RW", "STS"): 0.1,
        ("STS", "TC"): 0.1,
    }
    assert adoption_verdict(metrics, j) == "single"
