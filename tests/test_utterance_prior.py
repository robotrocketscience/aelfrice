"""Utterance-vs-knowledge document prior (#1174 item 3).

The lane ships inert, so the load-bearing tests here are the ones that pin
*which* class definition the table is built from and that the rerank is
byte-identical at W=0. Both were failure modes found by measurement, not
hypotheticals — see `src/aelfrice/utterance_prior.py`.
"""
from __future__ import annotations

import math

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.retrieval import _l1_hits, retrieve_v2
from aelfrice.store import MemoryStore
from aelfrice.utterance_prior import (
    CLASS_K_SOURCE_KINDS,
    CLASS_Q_SOURCE_KINDS,
    ENV_UTTERANCE_PRIOR_WEIGHT,
    UtterancePrior,
    resolve_utterance_prior_weight,
    utterance_logodds,
    utterance_prior_penalty,
)

UTTERANCES = [
    "can you go ahead and cut a new release for me",
    "what should we do about the failing test",
    "ok so lets run the bench and see what it says",
    "how do i turn the retrieval lane off",
    "please rebase this branch and push it",
    "should we ship this or wait for the review",
]
KNOWLEDGE = [
    "The retrieval budget defaults to 2000 tokens per injection.",
    "Beliefs are stored in a SQLite database under the git directory.",
    "The scorer applies per-field length normalisation before saturation.",
    "Locked beliefs are injected ahead of the ranked candidate list.",
    "Edge weights decay according to the configured half life.",
    "The migration runs once and records a sentinel on completion.",
]


# Each document is logged REPEATS times. `MIN_DOCUMENT_FREQUENCY` is 5, so a
# one-shot corpus builds an empty table and every assertion below would pass
# vacuously against a lane that does nothing. Repeating puts the shared stems
# over the production threshold, so these tests exercise the shipped config.
REPEATS = 5


def _seed(store: MemoryStore) -> None:
    """Populate the ingest log with both classes."""
    for rep in range(REPEATS):
        for i, text in enumerate(UTTERANCES):
            store.record_ingest(
                source_kind=CLASS_Q_SOURCE_KINDS[0],
                raw_text=text,
                source_path=None,
                raw_meta={"role": "user"},
                session_id=f"s{rep}-{i}",
            )
        for i, text in enumerate(KNOWLEDGE):
            store.record_ingest(
                source_kind=CLASS_K_SOURCE_KINDS[0],
                raw_text=text,
                source_path=f"src/mod{rep}_{i}.py",
                raw_meta=None,
                session_id=None,
            )


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(str(tmp_path / "m.db"))
    yield s
    s.close()


def test_class_sources_are_pinned() -> None:
    """The class definition is the whole ballgame (#1174 item 3).

    Training class Q on transcript-*derived beliefs* instead of the logged
    turns separates its own classes at AUC 0.908 while learning "is this text
    about source code" — measured on the live store, it ranked question-shaped
    prose BELOW statement-shaped prose (AUC 0.350), i.e. it would promote the
    echoes the lane exists to demote. If a future change widens or redefines
    these tuples, that measurement no longer holds and must be redone.
    """
    assert CLASS_Q_SOURCE_KINDS == ("transcript",)
    assert CLASS_K_SOURCE_KINDS == ("filesystem", "git")
    assert not set(CLASS_Q_SOURCE_KINDS) & set(CLASS_K_SOURCE_KINDS)


def test_prior_scores_utterances_above_knowledge(store: MemoryStore) -> None:
    _seed(store)
    prior = utterance_logodds(store)
    assert prior.n_utterance == len(UTTERANCES) * REPEATS
    assert prior.n_knowledge == len(KNOWLEDGE) * REPEATS
    assert prior.logodds, "expected a non-empty log-odds table"
    hi = max(prior.score(t) for t in KNOWLEDGE)
    lo = min(prior.score(t) for t in UTTERANCES)
    assert lo > hi, (
        "every utterance must outscore every knowledge document; "
        f"min utterance={lo:.4f} max knowledge={hi:.4f}"
    )


def test_empty_class_degrades_to_a_no_op(store: MemoryStore) -> None:
    """A store with only one class populated must score 0.0, not raise.

    Enabling the lane on a store with no ingest history is a configuration
    mistake, not a crash — and a log-odds against an empty class is undefined,
    so the table must simply be empty.
    """
    store.record_ingest(
        source_kind=CLASS_Q_SOURCE_KINDS[0],
        raw_text="only an utterance here",
        source_path=None,
        raw_meta=None,
        session_id="s",
    )
    prior = utterance_logodds(store)
    assert prior.logodds == {}
    assert prior.score("anything at all") == 0.0


def test_min_df_prunes_rare_stems(store: MemoryStore) -> None:
    """The document-frequency floor must actually drop vocabulary.

    Below it a stem's log-odds is dominated by smoothing, so it carries the
    class it happened to appear in rather than any evidence.
    """
    _seed(store)
    store.record_ingest(
        source_kind=CLASS_Q_SOURCE_KINDS[0],
        raw_text="zzqx wibblefrotz garrulousness",
        source_path=None,
        raw_meta=None,
        session_id="rare",
    )
    assert "zzqx" in utterance_logodds(store, min_df=1).logodds
    assert "zzqx" not in utterance_logodds(store).logodds


def test_unknown_vocabulary_is_neutral_not_utterance(store: MemoryStore) -> None:
    """Absence of evidence is not evidence of utterance."""
    _seed(store)
    prior = utterance_logodds(store)
    assert prior.score("zzqx wibblefrotz garrulousness") == 0.0


def test_penalty_is_a_pure_demotion(store: MemoryStore) -> None:
    """Knowledge-shaped content is left neutral, never promoted.

    The rerank score is log-domain and negative. An unclamped term would add
    a positive value for every knowledge document, reordering documents the
    lane has no opinion about.
    """
    _seed(store)
    prior = utterance_logodds(store)
    for text in KNOWLEDGE:
        assert utterance_prior_penalty(prior, text, 4.0) == 0.0
    for text in UTTERANCES:
        assert utterance_prior_penalty(prior, text, 4.0) < 0.0


def test_score_is_a_mean_not_a_sum(store: MemoryStore) -> None:
    """`score` must not scale with the number of distinct known stems.

    Length is already handled by BM25F's per-field normalisation, so
    summing here would double-count it — a long utterance would be
    demoted harder than a short one purely for carrying more vocabulary.
    Without this test, changing `sum(vals) / len(vals)` to `sum(vals)`
    leaves the whole file green.

    Asserted via the defining property of a mean: it is bounded by its
    parts. Concatenating two documents must land the score between their
    two individual scores; a sum lands outside that interval. Note the
    tokenizer takes a `set`, so repeating the *same* tokens proves
    nothing — the parts have to bring distinct vocabulary.
    """
    _seed(store)
    prior = utterance_logodds(store)
    a, b = UTTERANCES[0], UTTERANCES[1]
    sa, sb = prior.score(a), prior.score(b)
    both = prior.score(a + " " + b)
    assert min(sa, sb) <= both <= max(sa, sb), (
        f"score({a + ' ' + b}!r) = {both:.4f} fell outside "
        f"[{min(sa, sb):.4f}, {max(sa, sb):.4f}] — it is summing, not averaging"
    )
    # Guard the premise: if the two parts scored identically the interval
    # would be a point and the assertion above would be vacuous.
    assert sa != sb, "fixture parts score identically — test would be vacuous"


def test_penalty_scales_with_weight(store: MemoryStore) -> None:
    _seed(store)
    prior = utterance_logodds(store)
    text = UTTERANCES[0]
    p2 = utterance_prior_penalty(prior, text, 2.0)
    p4 = utterance_prior_penalty(prior, text, 4.0)
    assert p4 == pytest.approx(2.0 * p2)


def test_penalty_is_zero_when_lane_is_off(store: MemoryStore) -> None:
    _seed(store)
    prior = utterance_logodds(store)
    assert utterance_prior_penalty(prior, UTTERANCES[0], 0.0) == 0.0
    assert utterance_prior_penalty(None, UTTERANCES[0], 4.0) == 0.0


class TestWeightResolution:
    def test_defaults_to_zero(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_UTTERANCE_PRIOR_WEIGHT, raising=False)
        assert resolve_utterance_prior_weight() == 0.0

    def test_env_beats_explicit(self, monkeypatch) -> None:
        monkeypatch.setenv(ENV_UTTERANCE_PRIOR_WEIGHT, "8")
        assert resolve_utterance_prior_weight(2.0) == 8.0

    def test_explicit_used_without_env(self, monkeypatch) -> None:
        monkeypatch.delenv(ENV_UTTERANCE_PRIOR_WEIGHT, raising=False)
        assert resolve_utterance_prior_weight(4.0) == 4.0

    @pytest.mark.parametrize("raw", ["", "  ", "nonsense", "-1", "nan", "inf"])
    def test_unusable_env_falls_through(self, monkeypatch, raw: str) -> None:
        """A malformed or negative weight must not silently invert the lane
        into a promotion, and must not crash retrieval."""
        monkeypatch.setenv(ENV_UTTERANCE_PRIOR_WEIGHT, raw)
        assert resolve_utterance_prior_weight() == 0.0

    @pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan])
    def test_unusable_explicit_falls_through(self, monkeypatch, bad: float) -> None:
        monkeypatch.delenv(ENV_UTTERANCE_PRIOR_WEIGHT, raising=False)
        assert resolve_utterance_prior_weight(bad) == 0.0


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-07-30T00:00:00Z",
        last_retrieved_at=None,
    )


class TestRetrievalWiring:
    """The lane must be reachable from `retrieve_v2` and inert until asked."""

    @pytest.fixture()
    def populated(self, tmp_path):
        """A store with both ingest classes and one belief per document.

        A fixture rather than a helper so the SQLite connection is closed
        once per test instead of leaking across the file.
        """
        s = MemoryStore(str(tmp_path / "r.db"))
        _seed(s)
        for i, text in enumerate(UTTERANCES + KNOWLEDGE):
            s.insert_belief(_mk(f"b{i:02d}", text))
        yield s
        s.close()

    def test_off_is_byte_identical(self, populated, monkeypatch) -> None:
        monkeypatch.delenv(ENV_UTTERANCE_PRIOR_WEIGHT, raising=False)
        s = populated
        base = [b.id for b in retrieve_v2(s, "release bench", budget=4000).beliefs]
        zero = [
            b.id
            for b in retrieve_v2(
                s, "release bench", budget=4000, utterance_prior_weight=0.0,
            ).beliefs
        ]
        assert zero == base
        assert base, "fixture retrieved nothing — the test would be vacuous"

    def test_a_weight_reorders_utterances_downward(
        self, populated, monkeypatch,
    ) -> None:
        """The distinguishing assert: at a non-zero weight the ranking must
        actually change, and change in the demoting direction. A test that
        only checked 'does not raise' would pass against a no-op lane."""
        monkeypatch.delenv(ENV_UTTERANCE_PRIOR_WEIGHT, raising=False)
        s = populated
        query = "should we run the release bench"
        base = [b for b in retrieve_v2(s, query, budget=4000).beliefs]
        weighted = [
            b
            for b in retrieve_v2(
                s, query, budget=4000, utterance_prior_weight=8.0,
            ).beliefs
        ]
        assert [b.id for b in weighted] != [b.id for b in base], (
            "weight 8.0 left the ranking unchanged — the lane is not wired"
        )
        utter = set(UTTERANCES)
        base_rank = [i for i, b in enumerate(base) if b.content in utter]
        weighted_rank = [i for i, b in enumerate(weighted) if b.content in utter]
        if base_rank and weighted_rank:
            assert sum(weighted_rank) / len(weighted_rank) >= sum(base_rank) / len(
                base_rank
            ), "utterances must not move UP under a demotion weight"

    @pytest.mark.parametrize("bm25f", [True, False])
    def test_weight_survives_the_byte_identical_short_circuit(
        self, populated, monkeypatch, bm25f: bool,
    ) -> None:
        """Both lanes early-return, skipping the rerank loop, when every
        rerank input is off. Without the prior's own clause in those guards a
        caller who sets W there gets silence — the weight is accepted and
        then ignored.

        Driven through `_l1_hits` rather than `retrieve_v2` on purpose: the
        entity-persist lane resolves ON by default, which keeps `retrieve_v2`
        out of this branch entirely. The branch is still reachable in
        production whenever that lane is disabled, so it needs its own test.
        """
        monkeypatch.delenv(ENV_UTTERANCE_PRIOR_WEIGHT, raising=False)
        s = populated
        query = "the"
        common = {
            "l1_limit": 20,
            "posterior_weight": 0.0,
            "use_bm25f_anchors": bm25f,
            "use_entity_persist_demote": False,
            "use_supersession_demote": False,
        }
        plain = [b.id for b in _l1_hits(s, query, **common)]
        weighted = [
            b.id for b in _l1_hits(s, query, utterance_prior_weight=8.0, **common)
        ]
        assert len(plain) > 1, (
            "fixture returned <2 rows — a reorder assertion would be vacuous"
        )
        assert weighted != plain, (
            "W was swallowed by the byte-identical short-circuit"
        )

    def test_the_fts5_lane_applies_the_prior_too(
        self, populated, monkeypatch,
    ) -> None:
        """The rerank exists twice — once for BM25F, once for FTS5. Only the
        BM25F copy is on the default path, so the FTS5 copy needs its own
        test or it can be deleted without any test noticing.
        """
        monkeypatch.delenv(ENV_UTTERANCE_PRIOR_WEIGHT, raising=False)
        s = populated
        # FTS5 is implicit-AND, so a multi-word query matches only documents
        # containing every term and this fixture returns nothing for one. A
        # single common term is what keeps the lane non-empty here.
        query = "the"
        plain = [
            b.id
            for b in retrieve_v2(s, query, budget=4000, use_bm25f=False).beliefs
        ]
        weighted = [
            b.id
            for b in retrieve_v2(
                s, query, budget=4000, use_bm25f=False,
                utterance_prior_weight=8.0,
            ).beliefs
        ]
        assert len(plain) > 1, (
            "fixture returned <2 rows — a reorder assertion would be vacuous"
        )
        assert weighted != plain, "the FTS5 rerank ignored the prior"

    def test_prior_is_cached_on_the_store(self, populated, monkeypatch) -> None:
        """Building costs a full ingest-log pass; it must not run per query."""
        monkeypatch.delenv(ENV_UTTERANCE_PRIOR_WEIGHT, raising=False)
        s = populated
        assert getattr(s, "_utterance_prior_cache", None) is None
        retrieve_v2(s, "release", budget=4000, utterance_prior_weight=4.0)
        first = getattr(s, "_utterance_prior_cache", None)
        assert isinstance(first, UtterancePrior)
        retrieve_v2(s, "bench", budget=4000, utterance_prior_weight=4.0)
        assert getattr(s, "_utterance_prior_cache", None) is first

    def test_off_does_not_build_the_table(self, populated, monkeypatch) -> None:
        """At W=0 nothing may touch the ingest log."""
        monkeypatch.delenv(ENV_UTTERANCE_PRIOR_WEIGHT, raising=False)
        s = populated
        retrieve_v2(s, "release bench", budget=4000)
        assert getattr(s, "_utterance_prior_cache", None) is None
