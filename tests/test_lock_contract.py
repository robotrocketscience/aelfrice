"""Locked-belief contract acceptance tests for #379.

The contract (supersedes #373 + #199 H3):

- Locked beliefs (lock_level != LOCK_NONE) are the always-injected
  pool. SessionStart and UserPromptSubmit both emit every lock in
  full — no top-K, no scoring, no prompt-similarity gating.
- Top-K selection applies only to the non-locked retrieval surface
  (L1 / L2.5 / L3). The algorithm itself is OOS for #379; this file
  just verifies the contract boundary.
- A demoted lock (lock_level reset to LOCK_NONE) drops out of the
  always-injected pool and participates in normal retrieval.
"""
from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import pytest

from aelfrice.hook import session_start, user_prompt_submit
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore


def _mk(
    bid: str,
    content: str,
    lock_level: str = LOCK_NONE,
    locked_at: str | None = None,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=9.0 if lock_level == LOCK_USER else 1.0,
        beta=0.5 if lock_level == LOCK_USER else 1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=locked_at,
        created_at="2026-04-26T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(db: Path, beliefs: list[Belief]) -> None:
    store = MemoryStore(str(db))
    try:
        for b in beliefs:
            store.insert_belief(b)
    finally:
        store.close()


def _set_db(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setenv("AELFRICE_DB", str(path))


# ---- AC1: SessionStart injects every lock unconditionally --------------


def test_session_start_emits_every_lock_regardless_of_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All locks ship — lock count IS the operator's budget knob."""
    db = tmp_path / "memory.db"
    locks = [
        _mk(f"L{i}", f"locked truth number {i}", LOCK_USER,
            f"2026-04-26T00:00:{i:02d}Z")
        for i in range(20)
    ]
    _seed(db, locks)
    _set_db(monkeypatch, db)
    monkeypatch.chdir(tmp_path)

    sin = io.StringIO("{}")
    sout = io.StringIO()
    serr = io.StringIO()
    rc = session_start(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    out = sout.getvalue()
    for i in range(20):
        assert f'<belief id="L{i}"' in out, (
            f"L{i} missing from SessionStart payload — locks must "
            "always inject in full per #379"
        )


# ---- AC2: UPS includes every lock in addition to the prompt-driven pool


def test_retrieve_always_includes_every_lock_regardless_of_prompt(
    tmp_path: Path
) -> None:
    """retrieve() never trims locks; no prompt-overlap gate, no K-cap."""
    db = tmp_path / "memory.db"
    locks = [
        _mk(f"LOCK_{i}", f"deeply unrelated lock content alpha bravo {i}",
            LOCK_USER, f"2026-04-26T00:00:{i:02d}Z")
        for i in range(8)
    ]
    _seed(db, locks)

    store = MemoryStore(str(db))
    try:
        # Prompt has zero token overlap with any lock content.
        hits = retrieve(store, "completely orthogonal query terms zzzz")
        hit_ids = {h.id for h in hits}
        for i in range(8):
            assert f"LOCK_{i}" in hit_ids, (
                f"LOCK_{i} dropped from retrieve() under non-overlapping "
                "prompt — locks must always inject under #379"
            )
    finally:
        store.close()


def test_user_prompt_submit_block_contains_every_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: UPS hook payload always names each lock."""
    db = tmp_path / "memory.db"
    locks = [
        _mk(f"L{i}", f"lock body {i}", LOCK_USER,
            f"2026-04-26T00:00:{i:02d}Z")
        for i in range(5)
    ]
    _seed(db, locks)
    _set_db(monkeypatch, db)
    monkeypatch.chdir(tmp_path)

    sin = io.StringIO('{"prompt": "an entirely off-topic prompt"}')
    sout = io.StringIO()
    serr = io.StringIO()
    rc = user_prompt_submit(stdin=sin, stdout=sout, stderr=serr)
    assert rc == 0
    out = sout.getvalue()
    for i in range(5):
        assert f'id="L{i}"' in out, (
            f"L{i} missing from UPS payload — locks must always inject"
        )


# ---- AC4: demotion path drops a lock into the non-locked pool ----------


def test_demoted_lock_leaves_locked_pool_and_persists_in_store(
    tmp_path: Path
) -> None:
    """Reset lock_level to LOCK_NONE: belief stays, but is no longer locked.

    Verifies the demotion semantics #379 calls out: a pressured-out
    lock 'drops out of the always-injected pool back into the
    non-locked retrieval tail. They do NOT vanish.'
    """
    db = tmp_path / "memory.db"
    a = _mk("A", "stays locked", LOCK_USER, "2026-04-26T00:00:00Z")
    b = _mk("B", "soon to be demoted", LOCK_USER, "2026-04-26T00:00:01Z")
    _seed(db, [a, b])

    store = MemoryStore(str(db))
    try:
        before = {x.id for x in store.list_locked_beliefs()}
        assert before == {"A", "B"}

        # Demote B by resetting its lock_level. The store API requires a
        # full-row update_belief; we mutate the field on a fresh copy.
        b_demoted = replace(b, lock_level=LOCK_NONE, locked_at=None)
        store.update_belief(b_demoted)

        after = {x.id for x in store.list_locked_beliefs()}
        assert after == {"A"}, (
            "demoted belief should leave the locked pool"
        )

        # Belief still in store, just not in the always-inject set.
        all_ids = {row.id for row in store.find_orphan_beliefs()} | after
        # find_orphan_beliefs filters narrowly; instead probe by direct fetch.
        fetched = store.get_belief("B")
        assert fetched is not None, (
            "demoted belief must persist in store under #379 contract"
        )
        assert fetched.lock_level == LOCK_NONE
    finally:
        store.close()
