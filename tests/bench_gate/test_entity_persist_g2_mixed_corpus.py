"""G2 evidence eval for the entity-persistence demotion lane (#1096).

#1096 shipped the lane default-off with G1 (deterministic/byte-identical
off), G3 (offline AUC 0.48→0.87) and G4 (latency) met, but **G2 — the
demotion demonstrated on a retrieval eval without reducing durable
recall — was unmeasurable on the academic benches**: LoCoMo extracts
almost entirely ``noun_phrase`` entities, so every candidate takes the
same grounding-neutral penalty and nothing reorders (recall-safe but no
measurable demotion). This is the mixed-corpus retrieval eval that lane
needs, built as a self-contained deterministic fixture so G2 is a
repeatable public gate rather than a one-off lab measurement.

This eval cleared G2, and the lane subsequently flipped **default-ON in
``retrieve_v2``** at v4.0 (the resolver default; the legacy ``retrieve()``
path does not expose the lane — production cutover is tracked
separately). The tests below pass the flag explicitly, so they pin
behaviour on both sides of the default and are unaffected by the flip.

The corpus (``_entity_persist_mixed_store``) pairs, per topic, durable
technical beliefs (grounding to ``file_path`` / ``identifier`` /
``error_code``) with ephemeral coordination beliefs (grounding to
``branch`` / ``version``) that **share the query vocabulary** — so BM25
alone does not separate them (the control run interleaves ephemeral at
ranks 2–3), and the entity-persistence lane is the only signal that
sinks the coordination chatter.

G2 has two halves, both asserted here:

* **negative half (recall-safe):** durable recall must not drop when the
  lane is on — at any budget.
* **positive half (demotion):** ephemeral coordination hits must be
  demoted — at a realistic tight budget they fall out of the returned
  pack; the first durable answer's reciprocal rank rises.

Not corpus-gated: this runs on public CI (no ``bench_gated`` marker), so
it doubles as a regression lock on the lane's behaviour.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.retrieval import retrieve_v2
from tests.bench_gate._entity_persist_mixed_store import (
    TOPICS,
    build_mixed_store,
)

pytestmark = pytest.mark.timeout(120)

# Token budgets to sweep. 250 is the "tight pack" case where demotion
# pushes ephemeral OUT of the returned set; the larger budgets confirm
# recall-safety holds as the pack widens.
_BUDGETS = (250, 500, 1000)
# The tight budget at which the positive half (fall-out-of-pack) is
# asserted. At 250 the control returns all ephemeral; the lane must drop
# most of them.
_TIGHT_BUDGET = 250


def _ids(store, query: str, budget: int, *, demote: bool) -> list[str]:
    res = retrieve_v2(
        store, query, budget=budget, use_entity_persist_demote=demote
    )
    return [b.id for b in res.beliefs]


def _first_durable_rr(ids: list[str], durable: set[str]) -> float:
    """Reciprocal rank of the first durable belief (0.0 if none present)."""
    for rank, bid in enumerate(ids, start=1):
        if bid in durable:
            return 1.0 / rank
    return 0.0


@pytest.fixture()
def corpus(tmp_path: Path):
    c = build_mixed_store(tmp_path / "mixed.db")
    yield c
    c.store.close()


def test_corpus_grounds_as_labeled(corpus) -> None:
    """Guard against extractor drift: every durable belief must carry a
    high entity-persistence score and every ephemeral belief a low one,
    with neutral fillers absent from the persistence map entirely
    (#1098). If the extractor's kind assignment changes, the eval's
    premise breaks — fail here loudly rather than silently mis-measure.
    """
    store = corpus.store
    all_ids = sorted(corpus.durable_ids | corpus.ephemeral_ids)
    s1 = store.entity_persistence_scores(all_ids)

    for bid in sorted(corpus.durable_ids):
        assert bid in s1, f"durable {bid} missing from persistence map"
        assert s1[bid] >= 0.5, f"durable {bid} S1={s1[bid]} too low"

    for bid in sorted(corpus.ephemeral_ids):
        assert bid in s1, f"ephemeral {bid} missing from persistence map"
        assert s1[bid] < 0.25, f"ephemeral {bid} S1={s1[bid]} too high"

    # Neutral fillers extract only noun phrases → absent from the map,
    # so they are never demoted (#1098).
    neutral_scores = store.entity_persistence_scores(
        [f"neutral-{i}" for i in range(5)]
    )
    assert neutral_scores == {}, (
        f"neutral fillers must be absent from the map, got {neutral_scores}"
    )


def test_g2_recall_safe_and_demotes(corpus) -> None:
    """G2 on a mixed corpus: durable recall is preserved at every budget
    (negative half) AND ephemeral coordination hits are demoted at a
    tight budget (positive half). Prints the measured table for the
    #1096 flip decision.
    """
    store = corpus.store
    n_topics = len(TOPICS)

    print("\nentity-persist demotion — mixed-corpus G2 (#1096)")
    print(f"{'budget':>7} | {'durable off→on':>16} | "
          f"{'ephemeral off→on':>18} | {'MRR off→on':>16}")

    tight_eph_off = tight_eph_on = None
    for budget in _BUDGETS:
        dur_off = dur_on = eph_off = eph_on = 0
        mrr_off = mrr_on = 0.0
        for topic in TOPICS:
            label = corpus.labels[topic.query]
            durable = set(label["durable"])
            ephemeral = set(label["ephemeral"])

            off = _ids(store, topic.query, budget, demote=False)
            on = _ids(store, topic.query, budget, demote=True)

            dur_off += len(set(off) & durable)
            dur_on += len(set(on) & durable)
            eph_off += len(set(off) & ephemeral)
            eph_on += len(set(on) & ephemeral)
            mrr_off += _first_durable_rr(off, durable)
            mrr_on += _first_durable_rr(on, durable)

        print(f"{budget:>7} | {f'{dur_off}→{dur_on}':>16} | "
              f"{f'{eph_off}→{eph_on}':>18} | "
              f"{f'{mrr_off / n_topics:.3f}→{mrr_on / n_topics:.3f}':>16}")

        # --- G2 negative half: recall-safe at EVERY budget. ---
        assert dur_on >= dur_off, (
            f"budget={budget}: durable recall regressed {dur_off}→{dur_on}"
        )
        # --- MRR of the first durable answer must not drop. ---
        assert mrr_on + 1e-9 >= mrr_off, (
            f"budget={budget}: durable MRR regressed "
            f"{mrr_off / n_topics:.3f}→{mrr_on / n_topics:.3f}"
        )

        if budget == _TIGHT_BUDGET:
            tight_eph_off, tight_eph_on = eph_off, eph_on

    # --- G2 positive half: at the tight budget the lane demotes the
    # ephemeral coordination class substantially out of the pack. ---
    assert tight_eph_off is not None
    assert tight_eph_off > 0, "control must surface ephemeral to demote"
    assert tight_eph_on <= tight_eph_off // 2, (
        f"budget={_TIGHT_BUDGET}: lane failed to demote ephemeral "
        f"({tight_eph_off}→{tight_eph_on}; expected ≥50% reduction)"
    )


def test_default_on_after_flip(corpus) -> None:
    """Post-#1096 flip: the lane is default-ON in ``retrieve_v2``. A run
    that passes no flag matches an explicit ``demote=True`` run, and the
    opt-out (``demote=False``) stays reachable and reorders at least one
    topic — so the default flip is actually exercised, not a no-op. (G1's
    byte-identical-when-off invariant itself is pinned by the explicit
    ``demote=False`` arm in ``test_g2_recall_safe_and_demotes``.)"""
    store = corpus.store
    saw_difference = False
    for topic in TOPICS:
        explicit_on = _ids(store, topic.query, 1000, demote=True)
        explicit_off = _ids(store, topic.query, 1000, demote=False)
        default = [
            b.id
            for b in retrieve_v2(store, topic.query, budget=1000).beliefs
        ]
        assert default == explicit_on, (
            f"default should equal explicit-ON (flip landed) for {topic.query!r}"
        )
        if explicit_off != default:
            saw_difference = True
    assert saw_difference, (
        "the lane must reorder at least one topic vs the off-path — "
        "otherwise the default flip is untested"
    )
