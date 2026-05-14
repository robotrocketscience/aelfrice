"""Regression suite for the onboard rejection ledger (#801).

`aelf onboard <path>` historically re-emitted and re-classified every
sentence the host had previously rejected with `persist=False`, because
rejected sentences were never stored as beliefs and the dedup-by-id
filter only consulted `beliefs.id`. The fix persists `(belief_id, text,
source, rejected_at)` rows in `onboard_rejections` from
`accept_classifications`, and `start_onboard_session` +
`check_onboard_candidates` filter against that table.

Tests cover three layers:
- store-level CRUD on `onboard_rejections`
- accept_classifications writes / deletes ledger entries
- emit + check honor the ledger and the `force=True` bypass
- end-to-end: pass-2 emits 0 candidates after pass-1 rejects all
  (the issue's recorded repro: 723 rejects re-emitting forever)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.classification import (
    HostClassification,
    _derive_belief_id,
    accept_classifications,
    check_onboard_candidates,
    start_onboard_session,
)
from aelfrice.store import MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


def _populate_repo(root: Path) -> None:
    """Write content the three extractors find candidates in."""
    (root / "README.md").write_text(
        "This project must use uv for environment management.\n\n"
        "We always prefer atomic commits over batched commits.\n\n"
        "The system follows a Bayesian feedback loop with locks.\n"
    )
    (root / "module.py").write_text(
        '"""Top-level module docstring describing the module purpose."""\n\n'
        "def f():\n"
        '    """A top-level function that returns a constant value."""\n'
        "    return 1\n"
    )


# --- Store-level CRUD ---------------------------------------------------


def test_insert_then_is_rejected_returns_true(store: MemoryStore) -> None:
    store.insert_onboard_rejection(
        "aaaaaaaaaaaaaaaa", "noise sentence", "/x/y.py:1", "2026-05-14T00:00:00Z",
    )
    assert store.is_onboard_rejected("aaaaaaaaaaaaaaaa") is True


def test_is_rejected_unknown_id_returns_false(store: MemoryStore) -> None:
    assert store.is_onboard_rejected("ffffffffffffffff") is False


def test_insert_duplicate_id_is_idempotent(store: MemoryStore) -> None:
    store.insert_onboard_rejection(
        "aaaaaaaaaaaaaaaa", "noise", "/x.py:1", "2026-05-14T00:00:00Z",
    )
    store.insert_onboard_rejection(
        "aaaaaaaaaaaaaaaa", "noise", "/x.py:1", "2026-05-15T00:00:00Z",
    )
    assert store.count_onboard_rejections() == 1


def test_delete_removes_entry(store: MemoryStore) -> None:
    store.insert_onboard_rejection(
        "aaaaaaaaaaaaaaaa", "noise", "/x.py:1", "2026-05-14T00:00:00Z",
    )
    assert store.delete_onboard_rejection("aaaaaaaaaaaaaaaa") is True
    assert store.is_onboard_rejected("aaaaaaaaaaaaaaaa") is False


def test_delete_unknown_id_returns_false(store: MemoryStore) -> None:
    assert store.delete_onboard_rejection("ffffffffffffffff") is False


def test_list_rejection_ids_returns_set(store: MemoryStore) -> None:
    store.insert_onboard_rejection("aaaa1111aaaa1111", "x", "/a.py", "t")
    store.insert_onboard_rejection("bbbb2222bbbb2222", "y", "/b.py", "t")
    assert store.list_onboard_rejection_ids() == {
        "aaaa1111aaaa1111", "bbbb2222bbbb2222",
    }


def test_count_on_empty_store_is_zero(store: MemoryStore) -> None:
    assert store.count_onboard_rejections() == 0


# --- accept_classifications wires ledger writes -------------------------


def test_persist_false_writes_ledger_entry(
    store: MemoryStore, tmp_path: Path,
) -> None:
    _populate_repo(tmp_path)
    r = start_onboard_session(store, tmp_path, now="2026-05-14T00:00:00Z")
    assert len(r.sentences) > 0
    cls = [
        HostClassification(index=s.index, belief_type="factual", persist=False)
        for s in r.sentences
    ]
    accept_classifications(store, r.session_id, cls, now="2026-05-14T00:00:00Z")
    assert store.count_onboard_rejections() == len(r.sentences)


def test_persist_true_does_not_write_ledger_entry(
    store: MemoryStore, tmp_path: Path,
) -> None:
    _populate_repo(tmp_path)
    r = start_onboard_session(store, tmp_path, now="2026-05-14T00:00:00Z")
    cls = [
        HostClassification(index=s.index, belief_type="factual", persist=True)
        for s in r.sentences
    ]
    accept_classifications(store, r.session_id, cls, now="2026-05-14T00:00:00Z")
    assert store.count_onboard_rejections() == 0


def test_persist_true_after_prior_rejection_deletes_ledger_entry(
    store: MemoryStore, tmp_path: Path,
) -> None:
    _populate_repo(tmp_path)
    r1 = start_onboard_session(store, tmp_path, now="2026-05-14T00:00:00Z")
    cls_reject = [
        HostClassification(index=s.index, belief_type="factual", persist=False)
        for s in r1.sentences
    ]
    accept_classifications(store, r1.session_id, cls_reject)
    n_rejected = store.count_onboard_rejections()
    assert n_rejected > 0

    r2 = start_onboard_session(
        store, tmp_path, now="2026-05-14T01:00:00Z", force=True,
    )
    cls_accept = [
        HostClassification(index=s.index, belief_type="factual", persist=True)
        for s in r2.sentences
    ]
    accept_classifications(store, r2.session_id, cls_accept)
    assert store.count_onboard_rejections() == 0


# --- start_onboard_session honors the ledger ----------------------------


def test_second_pass_emits_zero_after_rejecting_all(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """The issue's recorded repro: 723 rejects re-emit forever today.

    With the ledger, pass 2 sees zero candidates after pass 1 rejects
    every sentence.
    """
    _populate_repo(tmp_path)
    r1 = start_onboard_session(store, tmp_path, now="2026-05-14T00:00:00Z")
    cls = [
        HostClassification(index=s.index, belief_type="factual", persist=False)
        for s in r1.sentences
    ]
    accept_classifications(store, r1.session_id, cls)

    r2 = start_onboard_session(store, tmp_path, now="2026-05-14T01:00:00Z")
    assert r2.sentences == []
    assert r2.n_already_rejected == len(r1.sentences)


def test_force_bypasses_ledger(store: MemoryStore, tmp_path: Path) -> None:
    _populate_repo(tmp_path)
    r1 = start_onboard_session(store, tmp_path, now="2026-05-14T00:00:00Z")
    cls = [
        HostClassification(index=s.index, belief_type="factual", persist=False)
        for s in r1.sentences
    ]
    accept_classifications(store, r1.session_id, cls)

    r2 = start_onboard_session(
        store, tmp_path, now="2026-05-14T01:00:00Z", force=True,
    )
    assert len(r2.sentences) == len(r1.sentences)
    assert r2.n_already_rejected == 0


# --- check_onboard_candidates honors the ledger -------------------------


def test_check_reports_already_rejected_count(
    store: MemoryStore, tmp_path: Path,
) -> None:
    _populate_repo(tmp_path)
    r1 = start_onboard_session(store, tmp_path, now="2026-05-14T00:00:00Z")
    cls = [
        HostClassification(index=s.index, belief_type="factual", persist=False)
        for s in r1.sentences
    ]
    accept_classifications(store, r1.session_id, cls)

    chk = check_onboard_candidates(store, tmp_path)
    assert chk.n_already_rejected == len(r1.sentences)
    assert chk.n_new == 0


def test_check_force_returns_rejected_to_new_bucket(
    store: MemoryStore, tmp_path: Path,
) -> None:
    _populate_repo(tmp_path)
    r1 = start_onboard_session(store, tmp_path, now="2026-05-14T00:00:00Z")
    cls = [
        HostClassification(index=s.index, belief_type="factual", persist=False)
        for s in r1.sentences
    ]
    accept_classifications(store, r1.session_id, cls)

    chk = check_onboard_candidates(store, tmp_path, force=True)
    assert chk.n_already_rejected == 0
    assert chk.n_new == len(r1.sentences)


def test_already_present_still_filtered_under_force(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """`force` opts back into the rejection ledger only — already-stored
    beliefs stay filtered even when `force=True`.
    """
    _populate_repo(tmp_path)
    r1 = start_onboard_session(store, tmp_path, now="2026-05-14T00:00:00Z")
    cls = [
        HostClassification(index=s.index, belief_type="factual", persist=True)
        for s in r1.sentences
    ]
    accept_classifications(store, r1.session_id, cls)

    chk = check_onboard_candidates(store, tmp_path, force=True)
    assert chk.n_new == 0
    assert chk.n_already_present > 0


# --- belief_id derivation sanity (key shared with beliefs.id) ------------


def test_rejection_belief_id_matches_derive_helper(
    store: MemoryStore, tmp_path: Path,
) -> None:
    """Ledger and dedup-by-id must share the same key derivation so the
    filter in start_onboard_session matches the writes from
    accept_classifications.
    """
    _populate_repo(tmp_path)
    r1 = start_onboard_session(store, tmp_path, now="2026-05-14T00:00:00Z")
    first = r1.sentences[0]
    expected_bid = _derive_belief_id(first.text, first.source)
    accept_classifications(
        store, r1.session_id,
        [HostClassification(index=first.index, belief_type="factual", persist=False)],
        now="2026-05-14T00:00:00Z",
    )
    assert store.is_onboard_rejected(expected_bid) is True
