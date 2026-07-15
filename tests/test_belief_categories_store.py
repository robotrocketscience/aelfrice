"""Store-layer tests for belief categories (#1126).

Covers the two additive tables and the CRUD + membership methods on
`MemoryStore`: upsert/get/list/set-trigger/delete categories,
assign/unassign membership, FK CASCADE, and the active-only member query.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.category import CategoryTrigger
from aelfrice.store import MemoryStore


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "cat.db"))


def _lock_belief(store: MemoryStore, text: str) -> str:
    """Insert a minimal user-locked belief and return its id."""
    from aelfrice.derivation import DerivationInput, derive
    from aelfrice.models import INGEST_SOURCE_CLI_REMEMBER

    d = derive(
        DerivationInput(
            raw_text=text,
            source_kind=INGEST_SOURCE_CLI_REMEMBER,
            ts="2026-07-15T00:00:00+00:00",
            session_id="s1",
        )
    )
    assert d.belief is not None
    store.insert_belief(d.belief)
    return d.belief.id


# --- category CRUD ------------------------------------------------------


def test_upsert_and_get_category(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        trig = CategoryTrigger(keywords=("push", "commit"))
        cat = store.upsert_category(
            name="git-workflow",
            always_on=False,
            trigger_json=trig.to_json(),
            default_lock="locked",
        )
        assert cat.name == "git-workflow"
        assert cat.always_on is False
        assert cat.trigger == trig
        assert cat.default_lock == "locked"
        assert cat.created_at  # stamped
        assert store.get_category("git-workflow") == cat
        assert store.get_category("nope") is None
    finally:
        store.close()


def test_upsert_is_idempotent_update(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        first = store.upsert_category(
            name="c", always_on=False, trigger_json="{}", default_lock="none"
        )
        second = store.upsert_category(
            name="c",
            always_on=True,
            trigger_json=CategoryTrigger(keywords=("x",)).to_json(),
            default_lock="locked",
        )
        assert second.always_on is True
        assert second.default_lock == "locked"
        assert second.trigger.keywords == ("x",)
        # created_at preserved across the update
        assert second.created_at == first.created_at
        # still one row
        assert len(store.list_categories()) == 1
    finally:
        store.close()


def test_list_categories_sorted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        for n in ("zeta", "alpha", "mid"):
            store.upsert_category(
                name=n, always_on=False, trigger_json="{}", default_lock="none"
            )
        assert [c.name for c in store.list_categories()] == ["alpha", "mid", "zeta"]
    finally:
        store.close()


def test_set_trigger_and_delete(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        store.upsert_category(
            name="c", always_on=False, trigger_json="{}", default_lock="none"
        )
        assert store.set_category_trigger(
            "c", CategoryTrigger(keywords=("y",)).to_json()
        )
        assert store.get_category("c").trigger.keywords == ("y",)  # type: ignore[union-attr]
        assert store.set_category_trigger("missing", "{}") is False
        assert store.delete_category("c") is True
        assert store.delete_category("c") is False
        assert store.get_category("c") is None
    finally:
        store.close()


# --- membership ---------------------------------------------------------


def test_assign_unassign_membership(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        bid = _lock_belief(store, "never push private content to a public remote")
        store.upsert_category(
            name="git-workflow",
            always_on=False,
            trigger_json="{}",
            default_lock="locked",
        )
        store.assign_belief_to_category(bid, "git-workflow")
        # idempotent
        store.assign_belief_to_category(bid, "git-workflow")
        assert store.get_categories_for_belief(bid) == ["git-workflow"]
        members = store.get_beliefs_for_category("git-workflow")
        assert [b.id for b in members] == [bid]

        assert store.unassign_belief_from_category(bid, "git-workflow") is True
        assert store.unassign_belief_from_category(bid, "git-workflow") is False
        assert store.get_categories_for_belief(bid) == []
    finally:
        store.close()


def test_assign_multi_membership(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        bid = _lock_belief(store, "run pre-commit before opening a PR")
        for n in ("repo-rules", "git-workflow"):
            store.upsert_category(
                name=n, always_on=False, trigger_json="{}", default_lock="locked"
            )
            store.assign_belief_to_category(bid, n)
        assert store.get_categories_for_belief(bid) == ["git-workflow", "repo-rules"]
    finally:
        store.close()


def test_assign_rejects_missing_belief_or_category(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        bid = _lock_belief(store, "some rule")
        store.upsert_category(
            name="c", always_on=False, trigger_json="{}", default_lock="none"
        )
        with pytest.raises(ValueError, match="no such belief"):
            store.assign_belief_to_category("deadbeef", "c")
        with pytest.raises(ValueError, match="no such category"):
            store.assign_belief_to_category(bid, "ghost")
    finally:
        store.close()


def test_delete_category_cascades_membership(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        bid = _lock_belief(store, "a rule")
        store.upsert_category(
            name="c", always_on=False, trigger_json="{}", default_lock="none"
        )
        store.assign_belief_to_category(bid, "c")
        store.delete_category("c")
        # membership row gone; belief survives
        assert store.get_categories_for_belief(bid) == []
        assert store.get_belief(bid) is not None
    finally:
        store.close()


def test_get_beliefs_for_category_excludes_retired(tmp_path: Path) -> None:
    store = _store(tmp_path)
    try:
        bid = _lock_belief(store, "a retiring rule")
        store.upsert_category(
            name="c", always_on=False, trigger_json="{}", default_lock="none"
        )
        store.assign_belief_to_category(bid, "c")
        b = store.get_belief(bid)
        assert b is not None
        b.valid_to = "2026-07-15T00:00:00+00:00"
        store.update_belief(b)
        # retired belief is excluded from the injection query
        assert store.get_beliefs_for_category("c") == []
        # but membership row still exists (curation-reversible)
        assert store.get_categories_for_belief(bid) == ["c"]
    finally:
        store.close()
