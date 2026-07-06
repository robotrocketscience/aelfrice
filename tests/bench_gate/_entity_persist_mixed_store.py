"""Synthetic mixed-corpus store for the entity-persistence demotion G2 eval (#1096).

The academic benches (LoCoMo) cannot exercise the entity-persistence
demotion lane: their corpora extract almost entirely ``noun_phrase``
entities, so every candidate takes the same grounding-neutral penalty
and nothing reorders — the lane is provably *recall-safe* there but its
*demotion* (G2 positive half) is unmeasurable. This module builds the
mixed corpus that lane needs: durable technical beliefs (grounding to
``file_path`` / ``identifier`` / ``error_code`` entities) and ephemeral
coordination beliefs (grounding to ``version`` / ``branch`` entities)
that **share query vocabulary**, so the two classes are genuine BM25
competitors for the same query.

Design rules that keep the eval honest (not rigged):

* Each topic's durable belief and ephemeral belief carry the SAME plain-
  English topic words (the query terms). The only systematic difference
  is *grounding*: durable content additionally names a file / symbol /
  error code; ephemeral content additionally names a branch / release
  version. So BM25 sees comparable lexical overlap for both, and the
  entity-persistence lane is the only signal that separates them.
* Ephemeral content contains NO durable entity (no file path, dotted or
  snake/Camel identifier, or error code) — otherwise its S1 would rise
  and the class label would be wrong. Content is checked against the
  real extractor in the companion test, so a drift in the extractor
  fails the fixture loudly.
* Fully deterministic and self-contained: no external corpus root, no
  randomness, no network. Public-safe — every string is invented.

The builder returns an on-disk ``MemoryStore`` plus the label sidecar
(gold durable ids / ephemeral ids per query) the eval scores against.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore


@dataclass(frozen=True)
class Topic:
    """One retrieval topic: a query, its durable gold answers, and the
    ephemeral coordination beliefs that share its vocabulary."""

    query: str
    durable: tuple[str, ...]   # belief contents that SHOULD win (high S1)
    ephemeral: tuple[str, ...]  # coordination distractors that SHOULD sink (S1~0)


# ---------------------------------------------------------------------------
# The corpus. Each topic shares plain-English terms across its durable and
# ephemeral beliefs; only the grounding differs.
# ---------------------------------------------------------------------------

TOPICS: tuple[Topic, ...] = (
    Topic(
        query="how does the retrieval ranker apply the demotion penalty",
        durable=(
            "The retrieval ranker applies the demotion penalty in "
            "retrieval.py through the apply_demotion_penalty helper.",
            "In the retrieval ranker the demotion penalty is a log term "
            "computed by score_candidate in ranker_core.py.",
        ),
        ephemeral=(
            "We shipped the retrieval ranker demotion penalty on branch "
            "feat/demotion-penalty for release v3.8.0.",
            "The retrieval ranker demotion penalty work merged in v3.8.0 "
            "after the review on branch fix/penalty-rounding.",
        ),
    ),
    Topic(
        query="what raises an error when the ingest budget is invalid",
        durable=(
            "The ingest path raises ValueError when the budget is invalid, "
            "guarded inside validate_budget in ingest_core.py.",
            "An invalid ingest budget triggers ValueError from the "
            "check_budget routine before any belief is written.",
        ),
        ephemeral=(
            "The invalid ingest budget error handling landed on branch "
            "fix/budget-guard for the v3.7.0 release.",
            "We discussed the invalid ingest budget error in the v3.7.0 "
            "cut and merged it from branch feat/budget-validation.",
        ),
    ),
    Topic(
        query="where is the session coherence filter implemented",
        durable=(
            "The session coherence filter lives in coherence.py and is "
            "driven by the session_coherence_score function.",
            "Session coherence filtering is implemented by the "
            "SessionCoherenceFilter class over the retrieval candidates.",
        ),
        ephemeral=(
            "The session coherence filter shipped on branch "
            "feat/session-coherence in release v3.5.0.",
            "We merged the session coherence filter for v3.5.0 off the "
            "branch fix/coherence-window after review.",
        ),
    ),
    Topic(
        query="how are corroboration counts recorded on re-ingest",
        durable=(
            "Corroboration counts are recorded by record_corroboration in "
            "store.py, one row per distinct re-assertion.",
            "On re-ingest the corroboration count increments through "
            "insert_or_corroborate without moving the belief posterior.",
        ),
        ephemeral=(
            "The corroboration count re-ingest change went out on branch "
            "feat/corroboration-dedup for release v3.6.0.",
            "We revisited corroboration count recording during the v3.6.0 "
            "cut and merged branch fix/corroboration-idempotent.",
        ),
    ),
    Topic(
        query="what computes the entity persistence grounding score",
        durable=(
            "The entity persistence grounding score is computed by "
            "entity_persistence_scores in store.py over the entity index.",
            "Entity persistence grounding is the S1 ratio produced by the "
            "compute_persistence helper from durable and transient counts.",
        ),
        ephemeral=(
            "The entity persistence grounding score landed on branch "
            "feat/entity-persist for release v3.8.0.",
            "We shipped the entity persistence grounding work in v3.8.0 "
            "from branch fix/persist-neutral after the review.",
        ),
    ),
    Topic(
        query="how does temporal decay affect retrieval ordering",
        durable=(
            "Temporal decay affects retrieval ordering via the "
            "apply_temporal_decay function in retrieval.py.",
            "Retrieval ordering under temporal decay is scaled by the "
            "TemporalDecayModel over each candidate age.",
        ),
        ephemeral=(
            "The temporal decay retrieval ordering change shipped on "
            "branch feat/temporal-decay for release v3.4.0.",
            "We merged the temporal decay retrieval ordering fix in "
            "v3.4.0 from branch fix/decay-halflife.",
        ),
    ),
    Topic(
        query="what handles a permission error during transcript capture",
        durable=(
            "Transcript capture handles PermissionError inside "
            "capture_turns in transcript_logger.py.",
            "A PermissionError during transcript capture is caught by the "
            "safe_write routine and logged without aborting the session.",
        ),
        ephemeral=(
            "The transcript capture permission error handling shipped on "
            "branch fix/capture-permission for release v3.6.0.",
            "We fixed the transcript capture permission error in the "
            "v3.6.0 cut off branch feat/capture-guard.",
        ),
    ),
    Topic(
        query="how is the belief posterior updated from feedback",
        durable=(
            "The belief posterior is updated from feedback by "
            "apply_feedback in feedback.py using a Bayesian update.",
            "Feedback moves the belief posterior through the "
            "bayesian_update helper on the alpha and beta counts.",
        ),
        ephemeral=(
            "The belief posterior feedback update change shipped on branch "
            "feat/feedback-posterior for release v3.2.0.",
            "We reworked the belief posterior feedback update in v3.2.0 "
            "off branch fix/posterior-clamp.",
        ),
    ),
    Topic(
        query="where does the lock level get enforced on retrieval",
        durable=(
            "The lock level is enforced on retrieval by the "
            "enforce_lock_level guard in retrieval.py.",
            "Retrieval respects the lock level through the "
            "LockLevelGate over the ranked candidates.",
        ),
        ephemeral=(
            "The lock level retrieval enforcement shipped on branch "
            "feat/lock-level for release v3.1.0.",
            "We landed the lock level retrieval enforcement in v3.1.0 "
            "from branch fix/lock-precedence.",
        ),
    ),
    Topic(
        query="how does the BM25 index build the augmented document",
        durable=(
            "The BM25 index builds the augmented document in bm25.py via "
            "the build_augmented_doc method over incoming anchor text.",
            "Augmented documents for the BM25 index come from the "
            "build_augmented_doc routine in bm25.py stitching anchor "
            "text onto each belief.",
        ),
        ephemeral=(
            "The BM25 augmented document change shipped on branch "
            "feat/bm25-augment for release v3.3.0.",
            "We merged the BM25 augmented document work in v3.3.0 off "
            "branch fix/anchor-weighting.",
        ),
    ),
)

# A few durable-but-entity-free filler beliefs (docstrings / formulae that
# extract only noun phrases) — the #1098 class that must be ABSENT from the
# persistence map and therefore never demoted. They give the corpus realism
# and let the eval confirm neutral content is untouched.
NEUTRAL_FILLERS: tuple[str, ...] = (
    "A retrieval budget bounds how many beliefs a single query returns.",
    "Corroboration measures how often the same claim is re-asserted.",
    "A demotion penalty lowers a candidate's rank without deleting it.",
    "Temporal decay reduces the weight of older evidence over time.",
    "Session coherence keeps a conversation's beliefs retrieved together.",
)


@dataclass
class MixedCorpus:
    """Handle returned by :func:`build_mixed_store`."""

    store: MemoryStore
    # query -> {"durable": [ids], "ephemeral": [ids]}
    labels: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    durable_ids: set[str] = field(default_factory=set)
    ephemeral_ids: set[str] = field(default_factory=set)


def _mk_belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-07-06T00:00:00Z",
        last_retrieved_at=None,
        session_id=None,
    )


def build_mixed_store(memory_db: Path) -> MixedCorpus:
    """Materialize the mixed corpus into an on-disk ``MemoryStore``.

    Entities auto-populate via ``insert_belief`` → ``_write_belief_entities``
    → the real ``extract_entities``; the corpus is therefore an end-to-end
    exercise of extraction + S1 + retrieval, not a hand-seeded entity table.

    Deterministic: same input path always yields the same rows/ids.
    Caller must ``store.close()`` when done.
    """
    store = MemoryStore(str(memory_db))
    corpus = MixedCorpus(store=store)

    for t_idx, topic in enumerate(TOPICS):
        dur_ids: list[str] = []
        eph_ids: list[str] = []
        for d_idx, content in enumerate(topic.durable):
            bid = f"dur-{t_idx:02d}-{d_idx}"
            store.insert_belief(_mk_belief(bid, content))
            dur_ids.append(bid)
            corpus.durable_ids.add(bid)
        for e_idx, content in enumerate(topic.ephemeral):
            bid = f"eph-{t_idx:02d}-{e_idx}"
            store.insert_belief(_mk_belief(bid, content))
            eph_ids.append(bid)
            corpus.ephemeral_ids.add(bid)
        corpus.labels[topic.query] = {"durable": dur_ids, "ephemeral": eph_ids}

    for n_idx, content in enumerate(NEUTRAL_FILLERS):
        store.insert_belief(_mk_belief(f"neutral-{n_idx}", content))

    return corpus
