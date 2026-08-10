"""Under truncation the fan lane changes the SET, not just the order (#1462).

`lookup_entities`' docstring and `docs/user/CONFIG.md` both told a consumer
that turning `fan_effect` on "only changes the ordering, so no consumer needs
to know which lane ran". That holds only while `limit` covers the whole
candidate pool. Truncate — a top-k, a token budget, any `[:n]` — and the two
branches take the top `limit` of *differently ordered* lists, so a caller gets
different beliefs depending on a flag it was told to ignore.

These tests pin the fact through the real path rather than asserting on prose,
because prose rots and a corrected sentence is not a gate. The fixture is the
minimal one that makes the point: every belief matches the ubiquitous entity
except one, which matches only the rare entity, so the two lanes order the pool
oppositely and the disagreement can come from nothing else.
"""
from __future__ import annotations

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
)
from aelfrice.store import MemoryStore

UBIQUITOUS = "src/common.py"
RARE = "src/widget.py"
N_FILLER = 12
#: aaa_common + zzz_rare + the fillers. The whole candidate pool.
POOL = N_FILLER + 2


def _mk(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"content for {bid}",
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-08-10T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_USER_TRANSCRIPT,
    )


def _entities(store: MemoryStore, bid: str, *lowers: str) -> None:
    store._conn.execute(  # pyright: ignore[reportPrivateUsage]
        "DELETE FROM belief_entities WHERE belief_id=?", (bid,)
    )
    for lower in lowers:
        store._conn.execute(  # pyright: ignore[reportPrivateUsage]
            "INSERT INTO belief_entities(belief_id, entity_lower, "
            "entity_raw, kind, span_start, span_end) VALUES (?,?,?,?,0,0)",
            (bid, lower, lower, "identifier"),
        )
    store._conn.commit()  # pyright: ignore[reportPrivateUsage]


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    # Everything matches one query entity, so overlap cannot separate the
    # candidates and the count lane falls through to id order. `zzz_rare`
    # therefore sorts LAST without fan and FIRST with it — the widest
    # possible disagreement, which is what makes truncation visible at
    # limit 1 rather than at some interior index.
    s.insert_belief(_mk("aaa_common"))
    _entities(s, "aaa_common", UBIQUITOUS)
    s.insert_belief(_mk("zzz_rare"))
    _entities(s, "zzz_rare", RARE)
    for i in range(N_FILLER):
        bid = f"filler_{i:02d}"
        s.insert_belief(_mk(bid))
        _entities(s, bid, UBIQUITOUS)
    yield s
    s.close()


def _ids(store: MemoryStore, *, fan: bool, limit: int) -> list[str]:
    return [
        bid
        for bid, _ in store.lookup_entities(
            [UBIQUITOUS, RARE], limit=limit, fan_effect=fan,
        )
    ]


def test_the_two_lanes_order_this_fixture_oppositely(store) -> None:
    """Control. Every assertion below is vacuous if the lanes agree here."""
    assert _ids(store, fan=False, limit=POOL)[-1] == "zzz_rare"
    assert _ids(store, fan=True, limit=POOL)[0] == "zzz_rare"


@pytest.mark.parametrize("limit", [1, 2, 3, 7])
def test_truncation_changes_the_set_not_only_the_order(store, limit) -> None:
    """The claim the documentation used to make, falsified.

    Counts match at every limit — that much of the old sentence was true,
    and it is why the row counter cannot be read as reorder evidence
    (#1434). The sets do not match, which is the half that was wrong.
    """
    off = _ids(store, fan=False, limit=limit)
    on = _ids(store, fan=True, limit=limit)

    assert len(off) == len(on) == limit
    assert set(off) != set(on), (
        f"at limit={limit} the lane changed only the order, "
        "so the documented caveat would not apply"
    )


def test_a_caller_that_takes_the_whole_pool_can_ignore_the_lane(store) -> None:
    """The converse, which is the actionable half of the caveat.

    Consume every returned row and the sets are identical — so the advice
    "take the full pool before slicing" is real advice, not a hedge.
    """
    off = _ids(store, fan=False, limit=POOL)
    on = _ids(store, fan=True, limit=POOL)

    assert set(off) == set(on)
    assert off != on  # ... and the ordering genuinely did change


def test_the_sets_converge_exactly_where_the_pool_is_covered(store) -> None:
    """Pins the boundary rather than a sampled point.

    Below the pool size the sets disagree; at and above it they agree. A
    test that only checked limit=1 would still pass if truncation stopped
    mattering at 2.
    """
    disagree = [
        n for n in range(1, POOL + 1)
        if set(_ids(store, fan=False, limit=n)) != set(_ids(store, fan=True, limit=n))
    ]
    assert disagree == list(range(1, POOL))
    assert set(_ids(store, fan=False, limit=POOL)) == set(
        _ids(store, fan=True, limit=POOL)
    )
