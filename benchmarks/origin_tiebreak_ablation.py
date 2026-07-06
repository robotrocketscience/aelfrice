"""#1089 axis-2 origin-priority tie-break — offline ablation (G3).

Builds a SYNTHETIC mixed-provenance corpus: for each query, one curated
belief (origin `user_validated`, as claude-memory user/feedback lands)
and one conversational belief (origin `user_transcript`, as passive
capture lands) share the query tokens so they TIE on the composite rerank
score. Reports how the tie-break moves the curated belief's rank.

The point it demonstrates: on a mixed-provenance store the tie-break
deterministically lifts curated content above conversational capture on a
tie, while leaving the flag-off ordering byte-identical. It also makes
the honest limitation visible — on a single-provenance corpus (e.g.
LoCoMo, whose beliefs are all conversational) every candidate shares an
origin tier, so the tie-break is INERT there and cannot regress recall.
That is why the default-ON flip is gated on a separate LoCoMo
no-regression run, not on this ablation.

SCOPE: the tie-break is wired into both ranked candidate tiers — L1 (FTS
rerank) and L2.5 (entity-index overlap) — so it applies whether a belief
is surfaced by keyword rerank or by an exact entity match. It is a pure
within-tier tie-break: it never reorders across tiers (L0 locked and the
L2.5 entity tier stay above L1 by construction) and never overrides the
relevance score, only breaks a genuine score/overlap tie.

No live-store content — deterministic and CI-safe.

Run: `python benchmarks/origin_tiebreak_ablation.py`
"""
from __future__ import annotations

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_USER_TRANSCRIPT,
    ORIGIN_USER_VALIDATED,
    Belief,
)
from aelfrice.retrieval import retrieve_v2
from aelfrice.store import MemoryStore

# Entity-free natural-language subjects, one unique pair per query so a
# query's tokens match exactly its own con/cur pair (top-2, tied) and no
# other. Kept plain-prose so retrieval routes through L1, not L2.5.
SUBJECTS = [
    "the planning cadence stays weekly on mondays",
    "standups happen right after the team lunch",
    "the roadmap review meeting is monthly",
    "product releases ship every other friday",
    "sprint retros close out each iteration",
    "the design critique runs on wednesdays",
    "oncall rotates between the backend folks",
    "the budget forecast updates each quarter",
    "customer calls cluster in the mornings",
    "the newsletter goes out on the first",
]


def _mk(bid: str, content: str, origin: str) -> Belief:
    return Belief(
        id=bid, content=content, content_hash=f"h_{bid}",
        alpha=1.0, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
        locked_at=None, created_at="2026-06-01T00:00:00Z",
        last_retrieved_at=None, origin=origin,
    )


def _rank_of_curated(store: MemoryStore, query: str, *, on: bool) -> int:
    ids = [b.id for b in retrieve_v2(store, query, use_origin_tiebreak=on).beliefs]
    for i, bid in enumerate(ids):
        if bid.startswith("cur_"):
            return i
    return len(ids)


def main() -> None:
    s = MemoryStore(":memory:")
    queries: list[str] = list(SUBJECTS)
    for i, subject in enumerate(SUBJECTS):
        # curated gets the LATER id so an id-only tie-break ranks it below
        # the conversational belief; only origin priority can lift it.
        s.insert_belief(_mk(f"con_{i}", subject, ORIGIN_USER_TRANSCRIPT))
        s.insert_belief(_mk(f"cur_{i}", subject, ORIGIN_USER_VALIDATED))
    s._conn.commit()

    off_ranks = [_rank_of_curated(s, q, on=False) for q in queries]
    on_ranks = [_rank_of_curated(s, q, on=True) for q in queries]
    s.close()

    n = len(queries)
    improved = sum(1 for o, n_ in zip(off_ranks, on_ranks) if n_ < o)
    mean_off = sum(off_ranks) / n
    mean_on = sum(on_ranks) / n
    print(f"synthetic mixed-provenance corpus: {n} tie queries (L1)")
    print(f"  mean curated rank OFF : {mean_off:.3f}")
    print(f"  mean curated rank ON  : {mean_on:.3f}")
    print(f"  queries where curated improved: {improved}/{n}")


if __name__ == "__main__":
    main()
