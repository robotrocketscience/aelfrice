"""FTS5 search truncates on a total order (#1370 §5/§6, #1157).

`search_beliefs` / `search_beliefs_scored` apply `LIMIT` to a
bm25-ranked scan. bm25 ties are ordinary — several beliefs carrying the
same terms score identically — so without a secondary sort key *which*
tied row survives the cut is decided by SQLite's scan order rather than
by the write log. These tests pin the tail to `bm25, b.id`.

The fixture is built so the two orders disagree: the beliefs share
identical content (guaranteeing an exact bm25 tie) and are inserted in
descending-id order, so rowid order is the reverse of id order. A
`limit` that cuts through the tie therefore returns the *last*-inserted
ids under the fix and the *first*-inserted ids without it.
"""
from __future__ import annotations

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore

# Six beliefs, one cut at three: enough that a scan-order truncation and
# an id-order truncation share no members at all.
_N_TIED = 6
_CUT = 3
_TIED_CONTENT = "widget calibration threshold rollout"


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
        created_at="2026-08-05T00:00:00Z",
        last_retrieved_at=None,
    )


def _tied_ids() -> list[str]:
    return [f"tie{i:02d}" for i in range(_N_TIED)]


def _tied_store() -> MemoryStore:
    s = MemoryStore(":memory:")
    # Reverse insertion order: rowid ascending == id descending.
    for bid in reversed(_tied_ids()):
        s.insert_belief(_mk(bid, _TIED_CONTENT))
    return s


def test_search_beliefs_cuts_a_bm25_tie_by_id() -> None:
    s = _tied_store()
    got = [b.id for b in s.search_beliefs(_TIED_CONTENT, limit=_CUT)]
    assert got == sorted(_tied_ids())[:_CUT]


def test_search_beliefs_scored_cuts_a_bm25_tie_by_id() -> None:
    s = _tied_store()
    got = [b.id for b, _score in s.search_beliefs_scored(_TIED_CONTENT, limit=_CUT)]
    assert got == sorted(_tied_ids())[:_CUT]


def test_tie_is_real_not_an_artifact_of_the_fixture() -> None:
    """Guard the fixture itself: all six rows must score identically.

    If the contents ever stop tying, the two tests above would pass for
    the wrong reason — bm25 alone would already order them.
    """
    s = _tied_store()
    scores = [score for _b, score in s.search_beliefs_scored(_TIED_CONTENT, limit=100)]
    assert len(scores) == _N_TIED
    assert len(set(scores)) == 1


def test_truncation_is_stable_under_reinsertion_order() -> None:
    """Same beliefs, opposite write order → same top-K.

    This is the property #1157 actually needs: the cut is a function of
    the belief set, not of the order rows happen to sit in the table.
    """
    forward = MemoryStore(":memory:")
    for bid in _tied_ids():
        forward.insert_belief(_mk(bid, _TIED_CONTENT))
    reverse = _tied_store()
    a = [b.id for b in forward.search_beliefs(_TIED_CONTENT, limit=_CUT)]
    b = [x.id for x in reverse.search_beliefs(_TIED_CONTENT, limit=_CUT)]
    assert a == b == sorted(_tied_ids())[:_CUT]
