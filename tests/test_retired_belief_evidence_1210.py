"""A retired belief must not gain evidence on any default path (#1210).

`get_belief` had no `valid_to` filter, so a soft-deleted belief stayed
reachable by id even though the FTS prune kept it out of search. The
measured consequence was that a retired belief accrued evidence: one
`aelf feedback` on a *neighbour* propagated into it, moving alpha and
writing an audit row, invisibly.

These tests are written against the *invariant* rather than the call
sites. The issue's own framing is the reason: there are ~60 `get_belief`
callers, and a suite that pins each one individually tells the next
reader nothing about whether the sixty-first is safe. What is asserted
here is that a retired belief's posterior cannot move, and that the
narrow set of callers that must address a tombstone still can.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.bfs_multihop import expand_bfs
from aelfrice.feedback import apply_feedback
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore

_KEEP = "B" + "1" * 15
_GONE = "B" + "2" * 15


def _belief(bid: str, content: str, *, alpha: float = 9.0) -> Belief:
    return Belief(
        id=bid, content=content, content_hash="h" + bid[1:],
        alpha=alpha, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
        locked_at=None, created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
    )


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    """The issue's reproduction: a retired belief with a live neighbour."""
    s = MemoryStore(str(tmp_path / "memory.db"))
    s.insert_belief(_belief(_KEEP, "deploy target is fly.io"))
    s.insert_belief(_belief(_GONE, "deploy target is heroku"))
    s.insert_edge(Edge(src=_KEEP, dst=_GONE, type="SUPPORTS", weight=1.0))
    s.soft_delete_belief(_GONE)
    yield s
    s.close()


# --- the filter itself ---------------------------------------------------


def test_get_belief_excludes_a_retired_belief(store: MemoryStore) -> None:
    assert store.get_belief(_GONE) is None
    assert store.get_belief(_KEEP) is not None


def test_include_retired_returns_the_tombstone(store: MemoryStore) -> None:
    """The opt-in is what audit and lifecycle callers use."""
    got = store.get_belief(_GONE, include_retired=True)
    assert got is not None
    assert got.valid_to is not None


def test_get_belief_in_scope_filters_the_local_branch(
    store: MemoryStore,
) -> None:
    assert store.get_belief_in_scope(_GONE, None) is None
    assert (
        store.get_belief_in_scope(_GONE, None, include_retired=True)
        is not None
    )


# --- the invariant -------------------------------------------------------


def test_feedback_on_a_neighbour_does_not_move_a_retired_posterior(
    store: MemoryStore,
) -> None:
    """The reproduction. Alpha went 9.0 -> 9.9 before #1210."""
    before = store.get_belief(_GONE, include_retired=True)
    assert before is not None

    apply_feedback(store, _KEEP, valence=1.0, source="cli")

    after = store.get_belief(_GONE, include_retired=True)
    assert after is not None
    assert after.alpha == before.alpha
    assert after.beta == before.beta


def test_feedback_on_a_neighbour_writes_no_audit_row_against_a_tombstone(
    store: MemoryStore,
) -> None:
    """One row was written before #1210, claiming evidence for a belief
    the user had retired — and invisible, because search could not
    surface it."""
    apply_feedback(store, _KEEP, valence=1.0, source="cli")
    rows = [e for e in store.list_feedback_events() if e.belief_id == _GONE]
    assert rows == []


def test_propagation_still_reaches_a_live_neighbour(
    tmp_path: Path,
) -> None:
    """The negative control for the two tests above.

    Without this, a bug that disabled propagation entirely would satisfy
    them both — they would pass on a store where nothing propagates to
    anything.
    """
    s = MemoryStore(str(tmp_path / "control.db"))
    try:
        s.insert_belief(_belief(_KEEP, "deploy target is fly.io"))
        s.insert_belief(_belief(_GONE, "deploy target is heroku"))
        s.insert_edge(Edge(src=_KEEP, dst=_GONE, type="SUPPORTS", weight=1.0))
        # Deliberately NOT retired.
        before = s.get_belief(_GONE)
        assert before is not None
        apply_feedback(s, _KEEP, valence=1.0, source="cli")
        after = s.get_belief(_GONE)
        assert after is not None
        assert after.alpha > before.alpha
        assert [e for e in s.list_feedback_events() if e.belief_id == _GONE]
    finally:
        s.close()


def test_bfs_does_not_surface_a_retired_belief(store: MemoryStore) -> None:
    seed = store.get_belief(_KEEP)
    assert seed is not None
    hops = expand_bfs([seed], store)
    assert _GONE not in [h.belief.id for h in hops]


# --- the callers that must still address a tombstone ---------------------


def test_insert_or_corroborate_does_not_collide_on_a_retired_id(
    store: MemoryStore,
) -> None:
    """A tombstone still owns its primary key.

    The id-collision guard in `insert_belief` opts in for this reason.
    Reverting that opt-in makes this raise `sqlite3.IntegrityError` on
    the UNIQUE constraint rather than corroborating.
    """
    clash = _belief(_GONE, "a different statement under the same id")
    clash.content_hash = "totally-different-hash"
    belief_id, was_inserted = store.insert_or_corroborate(
        clash, source_type="cli_remember",
    )
    assert belief_id == _GONE
    assert was_inserted is False
    # And the row is still the tombstone, not a resurrected copy.
    row = store.get_belief(_GONE, include_retired=True)
    assert row is not None
    assert row.valid_to is not None


def test_restore_returns_the_belief_at_the_posterior_it_was_retired_at(
    store: MemoryStore,
) -> None:
    """The end-to-end reason the invariant matters.

    Retirement is reversible by design; the defect made the state it
    restored to drift. Feedback lands on the neighbour while the belief
    is retired, and it comes back unchanged.
    """
    retired_alpha = store.get_belief(_GONE, include_retired=True).alpha
    apply_feedback(store, _KEEP, valence=1.0, source="cli")

    assert store.restore_belief(_GONE) is True
    restored = store.get_belief(_GONE)
    assert restored is not None
    assert restored.valid_to is None
    assert restored.alpha == retired_alpha
    # Back in keyword search, too.
    assert _GONE in [b.id for b in store.search_beliefs("heroku")]
