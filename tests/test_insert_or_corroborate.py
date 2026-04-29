"""Tests for MemoryStore.insert_or_corroborate.

Issue #219: content_hash dedup helper that prevents row inflation when the
same content arrives from different (source, sentence) pairs.
"""
from __future__ import annotations

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
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
        demotion_pressure=0,
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
    """Three duplicate hits from different sources accumulate three rows."""
    store = _fresh_store()
    try:
        b1 = _belief("id-001", "The sky is blue.", "hash-aaa")
        store.insert_belief(b1)

        for src in [
            CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
            CORROBORATION_SOURCE_COMMIT_INGEST,
            CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
        ]:
            b_dup = _belief("id-dup", "The sky is blue.", "hash-aaa")
            store.insert_or_corroborate(b_dup, source_type=src)

        assert store.count_corroborations("id-001") == 3
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
