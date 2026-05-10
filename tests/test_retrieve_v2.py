"""Tests for the lab-compatible retrieve_v2 wrapper."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import RetrievalResult, retrieve_v2
from aelfrice.store import MemoryStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(str(tmp_path / "rv2.db"))
    yield s
    s.close()


def _b(content: str, *, locked: bool = False, idx: int = 0) -> Belief:
    return Belief(
        id=f"b{idx}",
        content=content,
        content_hash=f"h{idx}",
        alpha=2.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_USER if locked else LOCK_NONE,
        locked_at="2026-04-27T00:00:00+00:00" if locked else None,
        demotion_pressure=0,
        created_at="2026-04-27T00:00:00+00:00",
        last_retrieved_at=None,
    )


def test_returns_retrieval_result_wrapper(store: MemoryStore) -> None:
    store.insert_belief(_b("The cat sat on the mat", idx=1))
    result = retrieve_v2(store, "cat")
    assert isinstance(result, RetrievalResult)
    assert isinstance(result.beliefs, list)


def test_budget_kwarg_routes_to_token_budget(store: MemoryStore) -> None:
    for i in range(10):
        store.insert_belief(_b(f"Token-priced belief number {i} content here", idx=i))
    full = retrieve_v2(store, "belief", budget=10_000)
    tight = retrieve_v2(store, "belief", budget=20)
    assert len(tight.beliefs) <= len(full.beliefs)


def test_include_locked_true_returns_locked(store: MemoryStore) -> None:
    store.insert_belief(_b("Locked truth statement here", locked=True, idx=1))
    store.insert_belief(_b("Unlocked corollary statement here", idx=2))
    result = retrieve_v2(store, "truth", budget=10_000, include_locked=True)
    assert any(b.lock_level == LOCK_USER for b in result.beliefs)


def test_include_locked_false_filters_locked(store: MemoryStore) -> None:
    store.insert_belief(_b("Locked truth statement here", locked=True, idx=1))
    store.insert_belief(_b("Unlocked truth corollary here", idx=2))
    result = retrieve_v2(store, "truth", budget=10_000, include_locked=False)
    assert all(b.lock_level == LOCK_NONE for b in result.beliefs)


def test_use_bfs_accepted_as_noop(store: MemoryStore) -> None:
    """``use_bfs`` round-trips with no observable ranking change on a
    single-belief store. Adapter call-site future-proofing.

    The deprecated ``use_hrr`` alias was removed alongside the
    vocabulary-bridge module in #536 — the structural lane (#152)
    is the production HRR surface and uses ``use_hrr_structural``."""
    store.insert_belief(_b("A factual statement to find", idx=1))
    a = retrieve_v2(store, "factual", use_bfs=False)
    b = retrieve_v2(store, "factual", use_bfs=True)
    assert [x.id for x in a.beliefs] == [x.id for x in b.beliefs]


def test_empty_query_returns_locked_only(store: MemoryStore) -> None:
    store.insert_belief(_b("Locked truth", locked=True, idx=1))
    store.insert_belief(_b("Unlocked search candidate text", idx=2))
    result = retrieve_v2(store, "", include_locked=True)
    assert all(b.lock_level == LOCK_USER for b in result.beliefs)


def test_diagnostic_fields_default_to_empty(store: MemoryStore) -> None:
    """Adapter code may read result.hrr_expansions / result.bfs_chains;
    those must exist as empty lists at v1.0.x to avoid AttributeError."""
    result = retrieve_v2(store, "anything")
    assert result.hrr_expansions == []
    assert result.bfs_chains == []
