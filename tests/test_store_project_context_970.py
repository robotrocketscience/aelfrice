"""Store-side project_context auto-stamp + backfill for #970.

Covers `insert_belief` stamping the store's repo identity
(`project_context_default`) onto eligible new beliefs, and the one-shot
`_maybe_backfill_project_context` pass over pre-existing rows. The
convention (chosen in #970): project_context holds repo identity;
project-scope non-user-locked rows get stamped; locked (L0) and
federation-shared rows stay '' (always-visible).
"""
from __future__ import annotations

from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SCOPE_GLOBAL,
    BELIEF_SCOPE_PROJECT,
    LOCK_NONE,
    LOCK_USER,
    Belief,
)
from aelfrice.store import (
    SCHEMA_META_PROJECT_CONTEXT_BACKFILL,
    MemoryStore,
)

_IDENTITY = "myrepo-deadbeef"


def _b(
    bid: str,
    *,
    project_context: str = "",
    scope: str = BELIEF_SCOPE_PROJECT,
    lock_level: str = LOCK_NONE,
) -> Belief:
    return Belief(
        id=bid,
        content=f"belief {bid}",
        content_hash=f"{bid}-hash",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        created_at="2026-06-18T00:00:00Z",
        last_retrieved_at=None,
        scope=scope,
        project_context=project_context,
    )


def _pc(store: MemoryStore, bid: str) -> str:
    b = store.get_belief(bid)
    assert b is not None
    return b.project_context


# --- insert-time stamping ------------------------------------------------

def test_insert_stamps_default_on_empty_project_scope() -> None:
    store = MemoryStore(":memory:", project_context_default=_IDENTITY)
    store.insert_belief(_b("b1"))
    assert _pc(store, "b1") == _IDENTITY


def test_insert_honours_explicit_project_context() -> None:
    store = MemoryStore(":memory:", project_context_default=_IDENTITY)
    store.insert_belief(_b("b1", project_context="explicit-slice"))
    assert _pc(store, "b1") == "explicit-slice"


def test_insert_leaves_user_locked_unstamped() -> None:
    store = MemoryStore(":memory:", project_context_default=_IDENTITY)
    store.insert_belief(_b("b1", lock_level=LOCK_USER))
    assert _pc(store, "b1") == ""


def test_insert_leaves_non_project_scope_unstamped() -> None:
    store = MemoryStore(":memory:", project_context_default=_IDENTITY)
    store.insert_belief(_b("b1", scope=BELIEF_SCOPE_GLOBAL))
    assert _pc(store, "b1") == ""


def test_insert_no_default_preserves_pre_970_behaviour() -> None:
    store = MemoryStore(":memory:")  # no repo identity → no stamping
    store.insert_belief(_b("b1"))
    assert _pc(store, "b1") == ""


# --- backfill ------------------------------------------------------------

def _seed_pre_970_rows(store: MemoryStore) -> None:
    """Insert rows that look like pre-#970 data: all project_context=''."""
    store.insert_belief(_b("plain"))
    store.insert_belief(_b("locked", lock_level=LOCK_USER))
    store.insert_belief(_b("global", scope=BELIEF_SCOPE_GLOBAL))


def test_backfill_stamps_only_eligible_rows() -> None:
    # Seed with no default so every row lands as ''.
    seed = MemoryStore(":memory:")
    _seed_pre_970_rows(seed)
    # Same connection, re-driven backfill with an identity present.
    seed._project_context_default = _IDENTITY  # pyright: ignore[reportPrivateUsage]
    stamped = seed._maybe_backfill_project_context()  # pyright: ignore[reportPrivateUsage]
    assert stamped == 1
    assert _pc(seed, "plain") == _IDENTITY
    assert _pc(seed, "locked") == ""
    assert _pc(seed, "global") == ""


def test_backfill_is_idempotent() -> None:
    seed = MemoryStore(":memory:")
    _seed_pre_970_rows(seed)
    seed._project_context_default = _IDENTITY  # pyright: ignore[reportPrivateUsage]
    first = seed._maybe_backfill_project_context()  # pyright: ignore[reportPrivateUsage]
    second = seed._maybe_backfill_project_context()  # pyright: ignore[reportPrivateUsage]
    assert first == 1
    assert second == 0
    assert seed.get_schema_meta(SCHEMA_META_PROJECT_CONTEXT_BACKFILL) is not None


def test_backfill_no_default_is_noop() -> None:
    seed = MemoryStore(":memory:")
    _seed_pre_970_rows(seed)
    assert seed._maybe_backfill_project_context() == 0  # pyright: ignore[reportPrivateUsage]
    assert seed.get_schema_meta(SCHEMA_META_PROJECT_CONTEXT_BACKFILL) is None
    assert _pc(seed, "plain") == ""


def test_backfill_does_not_touch_already_stamped_rows() -> None:
    store = MemoryStore(":memory:", project_context_default=_IDENTITY)
    store.insert_belief(_b("explicit", project_context="other-context"))
    # The on-open backfill already ran in __init__; explicit row untouched.
    assert _pc(store, "explicit") == "other-context"
