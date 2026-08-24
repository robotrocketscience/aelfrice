"""#1521 — the Stop candidate scan reads this session, not the whole store.

`_collect_lock_candidates` used to list every id in the store and fetch each
one, once per assistant turn, so a per-turn hook was linear in total store
size rather than in session size: 1.7 ms at 200 beliefs, 466 ms at 45,000.

The narrowing is only sound if it changes cost and nothing else, so the
equivalence test below is the load-bearing one — it compares the shipped
collector against a reference implementation of the old full walk over a
store seeded with every row class the SQL filter touches (other sessions,
retired rows, user-locked rows). The scan-width test is what fails if a
future edit quietly reverts to listing the whole store.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from aelfrice.hook import _belief_is_lock_candidate, _collect_lock_candidates
from aelfrice.models import (
    BELIEF_CORRECTION,
    BELIEF_FACTUAL,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_USER_TRANSCRIPT,
    Belief,
)
from aelfrice.store import MemoryStore

_SESSION = "s-1521"
_OTHER = "s-1521-other"
_CREATED = "2026-08-24T00:00:00Z"


def _belief(
    bid: str,
    content: str,
    *,
    session: str = _SESSION,
    type_: str = BELIEF_CORRECTION,
    lock: str = LOCK_NONE,
) -> Belief:
    return Belief(
        id=bid, content=content, content_hash=f"h_{bid}", alpha=1.0, beta=1.0,
        type=type_, lock_level=lock, locked_at=None, created_at=_CREATED,
        last_retrieved_at=None, session_id=session, origin=ORIGIN_USER_TRANSCRIPT,
    )


def _bid(seed: str) -> str:
    """Content-hash-shaped id, so no test passes on id ordering by accident."""
    return hashlib.sha256(f"1521\x00{seed}".encode()).hexdigest()[:16]


def _collect_via_full_walk(store: MemoryStore, session_id: str) -> list[Belief]:
    """Reference implementation: the pre-#1521 walk over every id."""
    out: list[Belief] = []
    for bid in store.list_belief_ids_newest_first():
        b = store.get_belief(bid)
        if b is None:
            continue
        if _belief_is_lock_candidate(b, session_id):
            out.append(b)
    return out


def _seeded_store(path: Path, *, mine: int, theirs: int) -> MemoryStore:
    """Store carrying every row class the narrowed query has to get right."""
    s = MemoryStore(str(path))
    for i in range(theirs):
        s.insert_belief(
            _belief(_bid(f"other-{i}"), f"Other session rule {i}.", session=_OTHER)
        )
    for i in range(mine):
        s.insert_belief(_belief(_bid(f"mine-{i}"), f"Always use rule {i}."))
    # Already user-locked: predicate rejects it, SQL must too.
    s.insert_belief(
        _belief(_bid("locked"), "Always lock this one.", lock=LOCK_USER)
    )
    # Not correction-class and not a directive: survives SQL, rejected in Python.
    s.insert_belief(
        _belief(_bid("plain"), "The sky was grey.", type_=BELIEF_FACTUAL)
    )
    # Retired: get_belief tombstones it, SQL must exclude it up front.
    retired = _bid("retired")
    s.insert_belief(_belief(retired, "Always use the retired rule."))
    s.soft_delete_belief(retired)
    return s


def test_narrowed_scan_matches_the_full_walk_exactly() -> None:
    """The load-bearing equivalence: same candidates, same order.

    Falsifiable by widening or narrowing the SQL predicate in
    `list_lock_candidate_ids` — dropping the lifecycle clause admits the
    tombstone, dropping the lock clause admits the locked row, and keying
    on `created_at` instead of `rowid` scrambles the order (every row here
    shares one timestamp, as real sessions do).
    """
    with tempfile.TemporaryDirectory() as td:
        s = _seeded_store(Path(td) / "m.db", mine=25, theirs=40)
        try:
            narrowed = _collect_lock_candidates(s, _SESSION)
            reference = _collect_via_full_walk(s, _SESSION)
        finally:
            s.close()

    assert [b.id for b in narrowed] == [b.id for b in reference]
    assert [b.content for b in narrowed] == [b.content for b in reference]
    # Adequacy: the fixture must actually exercise the filters.
    assert len(narrowed) == 25, "fixture no longer isolates the session"
    assert narrowed[0].content == "Always use rule 24.", "newest-first lost"


def test_the_scan_reads_only_this_session() -> None:
    """Cost contract: rows fetched scale with the session, not the store.

    This is the test that fails if `_collect_lock_candidates` is ever
    pointed back at `list_belief_ids_newest_first`.
    """
    with tempfile.TemporaryDirectory() as td:
        s = _seeded_store(Path(td) / "m.db", mine=10, theirs=500)
        # Bound before anything that can raise, so the `finally` restore
        # cannot mask a setup failure with an UnboundLocalError.
        real_get = s.get_belief
        try:
            listed = s.list_lock_candidate_ids(_SESSION)
            whole_store = s.list_belief_ids_newest_first()

            fetched: list[str] = []

            def counting_get(bid: str, **kw: object) -> Belief | None:
                fetched.append(bid)
                return real_get(bid, **kw)  # type: ignore[arg-type]

            s.get_belief = counting_get  # type: ignore[method-assign]
            candidates = _collect_lock_candidates(s, _SESSION)
        finally:
            s.get_belief = real_get  # type: ignore[method-assign]
            s.close()

    assert len(whole_store) == 513, "fixture size drifted"
    # 10 candidates + the user-locked row is excluded + the plain row survives
    # SQL and is rejected in Python + the tombstone is excluded.
    assert len(listed) == 11, listed
    assert len(fetched) == 11, "fetched more rows than the session holds"
    assert len(candidates) == 10
    # The whole point: reads did not scale with the 500 foreign rows.
    assert len(fetched) < len(whole_store) / 10


def test_the_listing_excludes_locked_retired_and_foreign_rows() -> None:
    with tempfile.TemporaryDirectory() as td:
        s = _seeded_store(Path(td) / "m.db", mine=3, theirs=3)
        try:
            listed = set(s.list_lock_candidate_ids(_SESSION))
        finally:
            s.close()

    assert _bid("locked") not in listed, "user-locked row survived the filter"
    assert _bid("retired") not in listed, "tombstone survived the filter"
    assert _bid("other-0") not in listed, "foreign session survived the filter"
    assert _bid("plain") in listed, (
        "SQL over-narrowed: classification belongs in Python, not the query"
    )


def test_an_empty_session_reads_nothing() -> None:
    with tempfile.TemporaryDirectory() as td:
        s = _seeded_store(Path(td) / "m.db", mine=0, theirs=20)
        try:
            assert s.list_lock_candidate_ids("s-1521-nonexistent") == []
            assert _collect_lock_candidates(s, "s-1521-nonexistent") == []
        finally:
            s.close()
