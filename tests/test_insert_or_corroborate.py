"""Tests for MemoryStore.insert_or_corroborate.

Issue #219: content_hash dedup helper that prevents row inflation when the
same content arrives from different (source, sentence) pairs.
"""
from __future__ import annotations

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    CORROBORATION_SOURCE_MCP_REMEMBER,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    CORROBORATION_SOURCES_NON_ASSERTING,
    LOCK_NONE,
    Belief,
)
from aelfrice.store import MemoryStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_store() -> MemoryStore:
    return MemoryStore(":memory:")


def _belief(
    bid: str,
    content: str,
    content_hash: str,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=content_hash,
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-04-28T00:00:00Z",
        last_retrieved_at=None,
    )


# ---------------------------------------------------------------------------
# Insert new belief
# ---------------------------------------------------------------------------


def test_insert_new_returns_id_and_true() -> None:
    """First insert returns (b.id, True) and adds one belief row."""
    store = _fresh_store()
    try:
        b = _belief("id-001", "The sky is blue.", "hash-aaa")
        belief_id, was_inserted = store.insert_or_corroborate(
            b, source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST
        )
        assert was_inserted is True
        assert belief_id == "id-001"
        assert store.get_belief("id-001") is not None
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Duplicate content_hash
# ---------------------------------------------------------------------------


def test_duplicate_hash_returns_existing_id_and_false() -> None:
    """Second call with same content_hash returns (existing_id, False)."""
    store = _fresh_store()
    try:
        b1 = _belief("id-001", "The sky is blue.", "hash-aaa")
        store.insert_belief(b1)

        # Different belief_id, same content_hash (different source).
        b2 = _belief("id-002", "The sky is blue.", "hash-aaa")
        belief_id, was_inserted = store.insert_or_corroborate(
            b2, source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST
        )
        assert was_inserted is False
        assert belief_id == "id-001"
        # Original row survives; no second row inserted.
        assert store.get_belief("id-002") is None
    finally:
        store.close()


def test_duplicate_hash_adds_corroboration_row() -> None:
    """Duplicate hit writes exactly one belief_corroborations row."""
    store = _fresh_store()
    try:
        b1 = _belief("id-001", "The sky is blue.", "hash-aaa")
        store.insert_belief(b1)

        assert store.count_corroborations("id-001") == 0

        b2 = _belief("id-002", "The sky is blue.", "hash-aaa")
        store.insert_or_corroborate(
            b2, source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST
        )

        assert store.count_corroborations("id-001") == 1
    finally:
        store.close()


def test_corroboration_count_increments_per_hit() -> None:
    """Hits from DISTINCT sources accumulate distinct rows; #1020 dedupes
    same-source repeats. Three distinct source_types -> three rows; a
    fourth repeat of an existing source_type adds nothing."""
    store = _fresh_store()
    try:
        b1 = _belief("id-001", "The sky is blue.", "hash-aaa")
        store.insert_belief(b1)

        for src in [
            CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
            CORROBORATION_SOURCE_COMMIT_INGEST,
            CORROBORATION_SOURCE_MCP_REMEMBER,
        ]:
            b_dup = _belief("id-dup", "The sky is blue.", "hash-aaa")
            store.insert_or_corroborate(b_dup, source_type=src)
        assert store.count_corroborations("id-001") == 3

        # #1020: a same-source repeat (same session/path/type) is ignored.
        b_dup = _belief("id-dup", "The sky is blue.", "hash-aaa")
        store.insert_or_corroborate(
            b_dup, source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST
        )
        assert store.count_corroborations("id-001") == 3
    finally:
        store.close()


def test_the_only_non_asserting_source_is_filesystem_ingest() -> None:
    """Ruled 2026-09-30: commits stay out, since each commit is its own event."""
    assert CORROBORATION_SOURCES_NON_ASSERTING == frozenset({
        CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    })


def test_distinct_commits_still_corroborate() -> None:
    store = _fresh_store()
    try:
        store.insert_belief(_belief("id-001", "The retrieval pipeline.", "hash-aaa"))
        for session in ("commit-1", "commit-2"):
            store.insert_or_corroborate(
                _belief("id-dup", "The retrieval pipeline.", "hash-aaa"),
                source_type=CORROBORATION_SOURCE_COMMIT_INGEST,
                session_id=session,
            )
        assert store.count_corroborations("id-001") == 2
    finally:
        store.close()


def test_non_asserting_sources_never_corroborate() -> None:
    """#1615: a file or commit re-read resolves to the belief, adds no row."""
    store = _fresh_store()
    try:
        store.insert_belief(_belief("id-001", "The sky is blue.", "hash-aaa"))
        for src in sorted(CORROBORATION_SOURCES_NON_ASSERTING):
            for session in ("scan-1", "scan-2", "scan-3"):
                bid, inserted = store.insert_or_corroborate(
                    _belief("id-dup", "The sky is blue.", "hash-aaa"),
                    source_type=src, session_id=session,
                )
                assert (bid, inserted) == ("id-001", False)
        assert store.count_corroborations("id-001") == 0
    finally:
        store.close()


def test_non_asserting_sources_still_insert_new_content() -> None:
    """The rule withholds corroboration, not ingestion."""
    store = _fresh_store()
    try:
        for i, src in enumerate(sorted(CORROBORATION_SOURCES_NON_ASSERTING)):
            b = _belief(f"id-new-{i}", f"New fact number {i}.", f"hash-new-{i}")
            assert store.insert_or_corroborate(b, source_type=src) == (b.id, True)
            assert store.get_belief(b.id) is not None
    finally:
        store.close()


def test_non_asserting_sources_skip_the_id_collision_row_too() -> None:
    """The #264 id-collision branch applies the same rule."""
    store = _fresh_store()
    try:
        store.insert_belief(_belief("id-001", "Original text.", "hash-aaa"))
        for src in sorted(CORROBORATION_SOURCES_NON_ASSERTING):
            # Same id, different content hash: only the id branch matches.
            bid, inserted = store.insert_or_corroborate(
                _belief("id-001", "Other text.", "hash-bbb"), source_type=src,
            )
            assert (bid, inserted) == ("id-001", False)
        assert store.count_corroborations("id-001") == 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Original belief unchanged on hit
# ---------------------------------------------------------------------------


def test_original_alpha_beta_unchanged_on_hit() -> None:
    """Hit path does not modify the canonical belief's alpha/beta."""
    store = _fresh_store()
    try:
        b1 = _belief("id-001", "The sky is blue.", "hash-aaa", alpha=2.0, beta=3.0)
        store.insert_belief(b1)

        b2 = _belief("id-002", "The sky is blue.", "hash-aaa", alpha=5.0, beta=1.0)
        store.insert_or_corroborate(
            b2, source_type=CORROBORATION_SOURCE_TRANSCRIPT_INGEST
        )

        canonical = store.get_belief("id-001")
        assert canonical is not None
        assert canonical.alpha == 2.0
        assert canonical.beta == 3.0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Bad source_type raises ValueError
# ---------------------------------------------------------------------------


def test_bad_source_type_raises_value_error() -> None:
    """Unknown source_type raises ValueError before touching the DB."""
    store = _fresh_store()
    try:
        b = _belief("id-001", "The sky is blue.", "hash-aaa")
        with pytest.raises(ValueError, match="Unknown source_type"):
            store.insert_or_corroborate(b, source_type="nonexistent_type")
    finally:
        store.close()


def test_bad_source_type_on_duplicate_raises_value_error() -> None:
    """Unknown source_type raises before reaching the corroboration insert."""
    store = _fresh_store()
    try:
        b1 = _belief("id-001", "The sky is blue.", "hash-aaa")
        store.insert_belief(b1)

        b2 = _belief("id-002", "The sky is blue.", "hash-aaa")
        with pytest.raises(ValueError, match="Unknown source_type"):
            store.insert_or_corroborate(b2, source_type="bad_type")

        # No corroboration row must exist after the failed call.
        assert store.count_corroborations("id-001") == 0
    finally:
        store.close()
