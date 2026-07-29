"""Regression: `aelf lock` must lock the RESOLVED belief, not the minted id.

`_lock_id(text)` keys on ``"lock\\0" + text`` but ``content_hash`` is a plain
sha256 of the text for *every* source. Text already ingested from a file,
transcript or commit therefore resolves to that pre-existing row, whose id is
not ``_lock_id(text)``. The lock-upgrade used to be gated on
``actual_id == lock_bid``, so the write was skipped and the CLI still printed
success — the product's single guarantee failing silently on its most natural
invocation (onboard a repo, then lock a rule you just read).

Falsifiable hypothesis: after `aelf lock` on text an earlier non-lock ingest
already stored, the resolved belief carries ``lock_level=user`` and appears in
``list_locked_beliefs()``.
"""
from __future__ import annotations

import argparse
import io
from collections.abc import Iterator
from pathlib import Path

import pytest

from aelfrice.cli import _cmd_lock
from aelfrice.derivation import DerivationInput, derive
from aelfrice.models import (
    INGEST_SOURCE_FILESYSTEM,
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_USER_STATED,
)
from aelfrice.store import MemoryStore

RULE = "Never push directly to the main branch; always use the publish script."


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[MemoryStore]:
    db = tmp_path / "lock-collision.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    s = MemoryStore(str(db))
    yield s
    s.close()


def _ns(statement: str) -> argparse.Namespace:
    return argparse.Namespace(statement=statement, session_id=None, doc_uri=None)


def _seed_from_filesystem(store: MemoryStore, text: str) -> str:
    """Insert `text` the way the repo scanner would, i.e. not via lock."""
    out = derive(
        DerivationInput(
            source_kind=INGEST_SOURCE_FILESYSTEM,
            raw_text=text,
            source_path="README.md",
            session_id=None,
            ts="2026-01-01T00:00:00+00:00",
        ),
    )
    assert out.belief is not None
    assert out.belief.lock_level == LOCK_NONE
    store.insert_or_corroborate(out.belief, source_type="filesystem_ingest")
    return out.belief.id


def test_lock_applies_to_pre_existing_non_lock_belief(
    store: MemoryStore,
) -> None:
    """The defect, directly: lock text the scanner already ingested."""
    seeded_id = _seed_from_filesystem(store, RULE)

    rc = _cmd_lock(_ns(RULE), io.StringIO())
    assert rc == 0

    resolved = store.get_belief(seeded_id)
    assert resolved is not None
    assert resolved.lock_level == LOCK_USER
    assert resolved.origin == ORIGIN_USER_STATED
    assert resolved.locked_at is not None


def test_locked_listing_includes_the_collided_belief(store: MemoryStore) -> None:
    """`aelf locked` must show it — the user-visible half of the bug."""
    seeded_id = _seed_from_filesystem(store, RULE)
    _cmd_lock(_ns(RULE), io.StringIO())

    assert seeded_id in {b.id for b in store.list_locked_beliefs()}


def test_lock_is_idempotent_on_the_resolved_id(store: MemoryStore) -> None:
    """Re-locking must not fork a second row or drop the lock."""
    seeded_id = _seed_from_filesystem(store, RULE)
    for _ in range(3):
        assert _cmd_lock(_ns(RULE), io.StringIO()) == 0

    locked = [b for b in store.list_locked_beliefs() if b.content == RULE]
    assert len(locked) == 1
    assert locked[0].id == seeded_id


def test_fresh_lock_still_works(store: MemoryStore) -> None:
    """Control: text with no prior ingest is unaffected by the change."""
    assert _cmd_lock(_ns("prefer uv over pip for every python task"), io.StringIO()) == 0

    locked = store.list_locked_beliefs()
    assert len(locked) == 1
    assert locked[0].lock_level == LOCK_USER
