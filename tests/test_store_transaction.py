"""#1135 write-group batching: `MemoryStore.transaction()`.

Contract under test: per-call commits inside the block are suppressed
and issued once at outermost exit; exceptions roll the whole group
back; invalidation callbacks fire once, after the commit; nested
blocks join the outermost transaction.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore


def _mk_belief(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"content for {bid}",
        content_hash="h_" + bid,
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-07-21T00:00:00Z",
        last_retrieved_at=None,
    )


def test_transaction_groups_writes_into_one_commit(tmp_path: Path) -> None:
    db = tmp_path / "txn.db"
    store = MemoryStore(str(db))
    try:
        with store.transaction():
            for i in range(5):
                store.insert_belief(_mk_belief(f"t{i}"))
            # Mid-transaction: a second connection must not see the
            # uncommitted rows (proves the inner commits were
            # suppressed, not just batched by chance).
            other = MemoryStore(str(db))
            try:
                assert other.count_beliefs() == 0
            finally:
                other.close()
        assert store.count_beliefs() == 5
        # Post-commit: the rows are durable for other connections.
        other = MemoryStore(str(db))
        try:
            assert other.count_beliefs() == 5
        finally:
            other.close()
    finally:
        store.close()


def test_transaction_rolls_back_on_exception(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "txn.db"))
    try:
        with pytest.raises(RuntimeError):
            with store.transaction():
                store.insert_belief(_mk_belief("doomed"))
                raise RuntimeError("boom")
        assert store.count_beliefs() == 0
        assert store.get_belief("doomed") is None
        # The store remains usable after the rollback.
        store.insert_belief(_mk_belief("after"))
        assert store.count_beliefs() == 1
    finally:
        store.close()


def test_invalidation_fires_once_after_commit(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "txn.db"))
    try:
        fired: list[int] = []
        store.add_invalidation_callback(
            lambda: fired.append(store.count_beliefs())
        )
        with store.transaction():
            store.insert_belief(_mk_belief("a"))
            store.insert_belief(_mk_belief("b"))
            assert fired == []  # deferred while the group is open
        # One post-commit fire; the callback observed committed state.
        assert fired == [2]
    finally:
        store.close()


def test_invalidation_not_fired_after_rollback(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "txn.db"))
    try:
        fired: list[bool] = []
        store.add_invalidation_callback(lambda: fired.append(True))
        with pytest.raises(RuntimeError):
            with store.transaction():
                store.insert_belief(_mk_belief("gone"))
                raise RuntimeError("boom")
        assert fired == []
    finally:
        store.close()


def test_nested_transactions_join_outermost(tmp_path: Path) -> None:
    db = tmp_path / "txn.db"
    store = MemoryStore(str(db))
    try:
        with store.transaction():
            store.insert_belief(_mk_belief("outer"))
            with store.transaction():
                store.insert_belief(_mk_belief("inner"))
            # Inner exit must NOT have committed yet.
            other = MemoryStore(str(db))
            try:
                assert other.count_beliefs() == 0
            finally:
                other.close()
        assert store.count_beliefs() == 2
    finally:
        store.close()


def test_mutations_outside_transaction_commit_per_call(
    tmp_path: Path,
) -> None:
    """The legacy path is unchanged: no transaction() means each
    mutating call commits immediately."""
    db = tmp_path / "txn.db"
    store = MemoryStore(str(db))
    try:
        store.insert_belief(_mk_belief("solo"))
        other = MemoryStore(str(db))
        try:
            assert other.count_beliefs() == 1
        finally:
            other.close()
    finally:
        store.close()
