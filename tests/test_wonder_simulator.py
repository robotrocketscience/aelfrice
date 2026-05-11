"""Unit tests for the synthetic corpus + feedback simulator (#228)."""
from __future__ import annotations

import random

import pytest

from aelfrice.store import MemoryStore
from aelfrice.wonder.simulator import (
    ALPHA_PROMOTION_THRESHOLD,
    SyntheticCorpus,
    build_corpus,
    feedback_verdict,
    populate_store,
    simulate_promotion,
)


def test_build_corpus_is_deterministic() -> None:
    a = build_corpus(rng=random.Random(0))
    b = build_corpus(rng=random.Random(0))
    assert [atom.belief_id for atom in a.atoms] == [
        atom.belief_id for atom in b.atoms
    ]
    assert [(e.src, e.dst, e.type) for e in a.edges] == [
        (e.src, e.dst, e.type) for e in b.edges
    ]


def test_build_corpus_default_size() -> None:
    c = build_corpus(rng=random.Random(0))
    # 8 topics × 25 atoms = 200, the spec's R0 size.
    assert len(c.atoms) == 200


def test_build_corpus_topic_assignment_is_partition() -> None:
    c = build_corpus(rng=random.Random(0), n_topics=4, n_atoms_per_topic=5)
    topics = {a.topic for a in c.atoms}
    assert topics == {0, 1, 2, 3}
    for topic in topics:
        in_topic = [a for a in c.atoms if a.topic == topic]
        assert len(in_topic) == 5


def test_build_corpus_rejects_too_few_atoms() -> None:
    with pytest.raises(ValueError):
        build_corpus(rng=random.Random(0), n_atoms_per_topic=2)


def test_build_corpus_rejects_too_few_topics() -> None:
    with pytest.raises(ValueError):
        build_corpus(rng=random.Random(0), n_topics=1)


def test_build_corpus_rejects_too_few_sessions() -> None:
    with pytest.raises(ValueError):
        build_corpus(rng=random.Random(0), n_sessions=1)


def test_populate_store_writes_all_atoms() -> None:
    store = MemoryStore(":memory:")
    corpus = build_corpus(rng=random.Random(0), n_topics=3, n_atoms_per_topic=5)
    populate_store(store, corpus, rng=random.Random(1))
    assert len(store.list_belief_ids()) == len(corpus.atoms)


def test_populate_store_stamps_high_uncertainty_some_atoms() -> None:
    store = MemoryStore(":memory:")
    corpus = build_corpus(rng=random.Random(0), n_topics=4, n_atoms_per_topic=10)
    populate_store(
        store, corpus, rng=random.Random(0),
        high_uncertainty_fraction=0.5,
    )
    high_count = 0
    for bid in store.list_belief_ids():
        b = store.get_belief(bid)
        assert b is not None
        if b.alpha < 1.0:
            high_count += 1
    assert high_count > 0


def test_feedback_verdict_same_topic_confirms() -> None:
    corpus = build_corpus(rng=random.Random(0), n_topics=3, n_atoms_per_topic=5)
    same_topic = [a.belief_id for a in corpus.atoms if a.topic == 0]
    assert feedback_verdict((same_topic[0], same_topic[1]), corpus) == "confirm"


def test_feedback_verdict_cross_topic_junks_when_signals_match() -> None:
    """Cross-topic composition junks under R1 only when both broadened
    paths are blocked — i.e. atoms share session AND share source_type."""
    from aelfrice.wonder.simulator import CorpusAtom, SyntheticCorpus
    corpus = SyntheticCorpus(
        atoms=(
            CorpusAtom("a0", 0, "sess_00", "commit_ingest"),
            CorpusAtom("a1", 1, "sess_00", "commit_ingest"),
        )
    )
    assert feedback_verdict(("a0", "a1"), corpus) == "junk"


def test_feedback_verdict_cross_topic_confirms_on_distinct_sources() -> None:
    """R1 rule 2 (#190 A1 signal): ≥2 source_types ⇒ confirm even cross-topic."""
    from aelfrice.wonder.simulator import CorpusAtom, SyntheticCorpus
    corpus = SyntheticCorpus(
        atoms=(
            CorpusAtom("a0", 0, "sess_00", "commit_ingest"),
            CorpusAtom("a1", 1, "sess_00", "transcript_ingest"),
        )
    )
    assert feedback_verdict(("a0", "a1"), corpus) == "confirm"


def test_feedback_verdict_cross_topic_confirms_on_distinct_sessions() -> None:
    """R1 rule 3 (#192 A2 signal): ≥2 session_ids ⇒ confirm even cross-topic."""
    from aelfrice.wonder.simulator import CorpusAtom, SyntheticCorpus
    corpus = SyntheticCorpus(
        atoms=(
            CorpusAtom("a0", 0, "sess_00", "commit_ingest"),
            CorpusAtom("a1", 1, "sess_01", "commit_ingest"),
        )
    )
    assert feedback_verdict(("a0", "a1"), corpus) == "confirm"


def test_feedback_verdict_unknown_belief_junks() -> None:
    corpus = build_corpus(rng=random.Random(0), n_topics=3, n_atoms_per_topic=5)
    assert feedback_verdict(("unknown_id",), corpus) == "junk"


def test_feedback_verdict_empty_composition_junks() -> None:
    corpus = SyntheticCorpus(atoms=())
    assert feedback_verdict((), corpus) == "junk"


def test_simulate_promotion_threshold_at_alpha_12() -> None:
    # initial α=1, +11 confirms → α=12 → promotes
    assert simulate_promotion(11, 0)
    assert not simulate_promotion(10, 0)
    assert ALPHA_PROMOTION_THRESHOLD == 12.0


def test_simulate_promotion_with_junks() -> None:
    # Junks accrue β but don't affect the α-only gate.
    assert simulate_promotion(11, 100)
