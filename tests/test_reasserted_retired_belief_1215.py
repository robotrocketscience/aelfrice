"""Re-asserting a retired statement has a defined outcome (#1215).

`get_belief_by_content_hash` had no `valid_to` filter, so a re-assertion
of retired content resolved to the **tombstone**: no row was inserted, a
corroboration row was written against the retired belief, and nothing
became visible. Via `aelf lock` the command printed success while `aelf
locked` stayed empty — the user's most explicit assertion of ground truth
was a no-op on every retrieval surface.

The ratified policy is tiered by who is asserting. A person (`aelf lock`,
`aelf remember`, and the MCP twins) revives the belief; background
capture (transcript, commit, filesystem, wonder, migration) leaves the
tombstone retired and records nothing.

These tests are written against that invariant. Each tier carries a
negative control on live content, because "background capture did not
revive" and "background capture did nothing at all" are satisfied by the
same assertions on a store where corroboration is broken outright.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_CLI_REMEMBER,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    CORROBORATION_SOURCE_MCP_REMEMBER,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    CORROBORATION_SOURCE_WONDER_INGEST,
    CORROBORATION_SOURCES_USER_EXPLICIT,
    CORROBORATION_SOURCE_TYPES,
    FEEDBACK_SOURCE_REASSERT_REVIVE,
    LOCK_NONE,
    Belief,
)
from aelfrice.store import MemoryStore
from aelfrice.wonder.lifecycle import Phantom, wonder_ingest

_ORIGINAL = "B" + "1" * 15
_REASSERT = "B" + "2" * 15
_HASH = "shared-content-hash"

_CAPTURE_SOURCES = [
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    CORROBORATION_SOURCE_WONDER_INGEST,
]


def _belief(
    bid: str, *, alpha: float = 9.0, content_hash: str = _HASH,
) -> Belief:
    return Belief(
        id=bid, content="deploy target is heroku", content_hash=content_hash,
        alpha=alpha, beta=1.0, type=BELIEF_FACTUAL, lock_level=LOCK_NONE,
        locked_at=None, created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
    )


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    """A retired belief, and a re-assertion of its content pending."""
    s = MemoryStore(str(tmp_path / "memory.db"))
    s.insert_belief(_belief(_ORIGINAL))
    s.soft_delete_belief(_ORIGINAL)
    yield s
    s.close()


# --- the lookup ----------------------------------------------------------


def test_content_hash_lookup_excludes_a_retired_belief(
    store: MemoryStore,
) -> None:
    assert store.get_belief_by_content_hash(_HASH) is None


def test_content_hash_lookup_opt_in_returns_the_tombstone(
    store: MemoryStore,
) -> None:
    """The UNIQUE-constraint guard needs to see it (#219)."""
    got = store.get_belief_by_content_hash(_HASH, include_retired=True)
    assert got is not None
    assert got.id == _ORIGINAL
    assert got.valid_to is not None


# --- background capture must not undo curation ---------------------------


@pytest.mark.parametrize("source", _CAPTURE_SOURCES)
def test_capture_does_not_revive_a_retired_belief(
    store: MemoryStore, source: str,
) -> None:
    belief_id, was_inserted = store.insert_or_corroborate(
        _belief(_REASSERT), source_type=source,
    )
    assert (belief_id, was_inserted) == (_ORIGINAL, False)
    row = store.get_belief(_ORIGINAL, include_retired=True)
    assert row is not None
    assert row.valid_to is not None
    assert store.get_belief(_ORIGINAL) is None
    assert store.search_beliefs("heroku") == []


@pytest.mark.parametrize("source", _CAPTURE_SOURCES)
def test_capture_writes_no_corroboration_against_a_tombstone(
    store: MemoryStore, source: str,
) -> None:
    store.insert_or_corroborate(_belief(_REASSERT), source_type=source)
    row = store.get_belief(_ORIGINAL, include_retired=True)
    assert row is not None
    assert row.corroboration_count == 0


@pytest.mark.parametrize("source", _CAPTURE_SOURCES)
def test_capture_still_corroborates_live_content(
    tmp_path: Path, source: str,
) -> None:
    """Negative control for the two above.

    Without it, a bug that made `insert_or_corroborate` a no-op for these
    sources would satisfy both — nothing revived, nothing corroborated,
    on a store where capture records nothing at all.
    """
    s = MemoryStore(str(tmp_path / f"control-{source}.db"))
    try:
        s.insert_belief(_belief(_ORIGINAL))  # deliberately NOT retired
        belief_id, was_inserted = s.insert_or_corroborate(
            _belief(_REASSERT), source_type=source,
        )
        assert (belief_id, was_inserted) == (_ORIGINAL, False)
        row = s.get_belief(_ORIGINAL)
        assert row is not None
        assert row.corroboration_count == 1
    finally:
        s.close()


# --- a person's re-assertion revives -------------------------------------


@pytest.mark.parametrize("source", sorted(CORROBORATION_SOURCES_USER_EXPLICIT))
def test_user_assertion_revives_a_retired_belief(
    store: MemoryStore, source: str,
) -> None:
    belief_id, was_inserted = store.insert_or_corroborate(
        _belief(_REASSERT), source_type=source,
    )
    assert (belief_id, was_inserted) == (_ORIGINAL, False)
    row = store.get_belief(_ORIGINAL)
    assert row is not None
    assert row.valid_to is None
    assert row.corroboration_count == 1
    # Back in keyword search, which is the surface the defect hid behind.
    assert [b.id for b in store.search_beliefs("heroku")] == [_ORIGINAL]


@pytest.mark.parametrize("source", sorted(CORROBORATION_SOURCES_USER_EXPLICIT))
def test_revival_preserves_the_posterior_it_was_retired_at(
    store: MemoryStore, source: str,
) -> None:
    """Revival is not evidence. The belief comes back where it left."""
    before = store.get_belief(_ORIGINAL, include_retired=True)
    assert before is not None
    store.insert_or_corroborate(_belief(_REASSERT), source_type=source)
    after = store.get_belief(_ORIGINAL)
    assert after is not None
    assert (after.alpha, after.beta) == (before.alpha, before.beta)


@pytest.mark.parametrize("source", sorted(CORROBORATION_SOURCES_USER_EXPLICIT))
def test_revival_leaves_an_audit_row(store: MemoryStore, source: str) -> None:
    """The transition must not be silent in either direction."""
    store.insert_or_corroborate(_belief(_REASSERT), source_type=source)
    sources = [
        e.source for e in store.list_feedback_events()
        if e.belief_id == _ORIGINAL
    ]
    assert FEEDBACK_SOURCE_REASSERT_REVIVE in sources


def test_a_live_belief_is_not_touched_by_the_revival_path(
    tmp_path: Path,
) -> None:
    """Negative control: no audit row when nothing was retired."""
    s = MemoryStore(str(tmp_path / "live.db"))
    try:
        s.insert_belief(_belief(_ORIGINAL))
        s.insert_or_corroborate(
            _belief(_REASSERT), source_type=CORROBORATION_SOURCE_CLI_REMEMBER,
        )
        sources = [
            e.source for e in s.list_feedback_events()
            if e.belief_id == _ORIGINAL
        ]
        assert FEEDBACK_SOURCE_REASSERT_REVIVE not in sources
    finally:
        s.close()


# --- the tier table itself -----------------------------------------------


def test_user_explicit_sources_are_a_subset_of_the_known_sources() -> None:
    """A typo here would silently demote a user path to capture."""
    assert CORROBORATION_SOURCES_USER_EXPLICIT <= CORROBORATION_SOURCE_TYPES
    assert CORROBORATION_SOURCES_USER_EXPLICIT == {
        CORROBORATION_SOURCE_CLI_REMEMBER,
        CORROBORATION_SOURCE_MCP_REMEMBER,
    }


# --- the other UNIQUE-constraint guard -----------------------------------


def test_wonder_ingest_does_not_regenerate_a_retired_phantom(
    tmp_path: Path,
) -> None:
    """`wonder_ingest` dedupes on a synthetic constituent-set hash.

    A GC'd phantom still owns that hash, so without the opt-in the skip
    check misses it and the insert trips `UNIQUE(content_hash)`. Skipping
    is also right on its own terms: a phantom the lifecycle retired
    should not come back on the next pass.
    """
    left, right = "B" + "7" * 15, "B" + "8" * 15
    s = MemoryStore(str(tmp_path / "wonder.db"))
    try:
        s.insert_belief(_belief(left))
        s.insert_belief(_belief(right, content_hash="other-hash"))
        phantom = Phantom(
            constituent_belief_ids=(left, right),
            generator="bfs",
            content="a speculative bridge",
            score=0.5,
        )
        first = wonder_ingest(s, [phantom])
        assert first.inserted == 1
        [phantom_id] = [
            bid for bid in s.list_belief_ids() if bid not in (left, right)
        ]
        s.soft_delete_belief(phantom_id)

        second = wonder_ingest(s, [phantom])
        assert second.inserted == 0
        assert second.skipped == 1
        row = s.get_belief(phantom_id, include_retired=True)
        assert row is not None
        assert row.valid_to is not None
    finally:
        s.close()
