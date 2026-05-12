"""Tests for the v3.0 #644 speculative_hash_v2 backfill migration.

Covers the ``MemoryStore._maybe_rehash_speculative_v2`` one-shot pass
that re-derives ``content_hash`` for every existing speculative belief
using the new (constituent_set, generator) key.

The migration is exercised by:

1. Inserting a phantom under the live (v2) ``wonder_ingest`` path — the
   row already carries a v2 hash.
2. Reverting that row's ``content_hash`` to the v1 layout by manually
   computing what the v1 algorithm would have produced.
3. Dropping the ``SCHEMA_META_SPECULATIVE_HASH_V2_COMPLETE`` marker.
4. Calling ``_maybe_rehash_speculative_v2()`` directly (or re-opening
   the store) and asserting the row's hash is back at v2.

This shape avoids having to build a fully synthetic pre-#644 DB; it
exercises the algorithm against a real-shaped row.
"""
from __future__ import annotations

import hashlib

from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SPECULATIVE,
    LOCK_NONE,
    ORIGIN_SPECULATIVE,
    RETENTION_FACT,
    Belief,
    Phantom,
)
from aelfrice.store import (
    SCHEMA_META_SPECULATIVE_HASH_V2_COMPLETE,
    MemoryStore,
)
from aelfrice.wonder.lifecycle import wonder_ingest


def _v1_hash(constituent_ids: tuple[str, ...]) -> str:
    """Reproduce the v1 _constituent_key format for migration testing."""
    raw = "wonder_ingest:" + ":".join(sorted(constituent_ids))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _constituent(bid: str) -> Belief:
    return Belief(
        id=bid,
        content=f"constituent {bid}",
        content_hash=f"ch_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-01T00:00:00+00:00",
        last_retrieved_at=None,
        retention_class=RETENTION_FACT,
    )


def _seed_v1_shaped_phantom(
    store: MemoryStore,
    *,
    constituents: tuple[str, ...] = ("a", "b"),
    generator: str = "subagent_dispatch:axis_A",
) -> tuple[str, str, str]:
    """Insert a phantom and revert its content_hash to the v1 layout.

    Returns ``(phantom_id, v1_hash, v2_hash)`` so the caller can locate
    the row and assert the migration moved its hash from v1 to v2.
    The phantom's ULID is kept as-is to avoid foreign-key churn on
    edges / corroborations.
    """
    for cid in constituents:
        store.insert_belief(_constituent(cid))

    wonder_ingest(
        store,
        [Phantom(
            constituent_belief_ids=constituents,
            generator=generator,
            content="phantom content",
            score=0.75,
        )],
    )

    cur = store._conn.execute(
        "SELECT id, content_hash FROM beliefs WHERE origin = ?",
        (ORIGIN_SPECULATIVE,),
    )
    rows = cur.fetchall()
    assert len(rows) == 1, f"expected 1 phantom, got {len(rows)}"
    phantom_id = str(rows[0]["id"])
    v2_hash = str(rows[0]["content_hash"])

    v1_hash = _v1_hash(constituents)
    store._conn.execute(
        "UPDATE beliefs SET content_hash = ? WHERE id = ?",
        (v1_hash, phantom_id),
    )
    store._conn.commit()
    return phantom_id, v1_hash, v2_hash


def test_migration_rehashes_v1_speculative_row(tmp_path) -> None:
    """A v1-shaped speculative row is rehashed to v2 on next open."""
    db = str(tmp_path / "spec_v1.db")
    s = MemoryStore(db)
    phantom_id, v1_hash, v2_hash = _seed_v1_shaped_phantom(s)

    got = s._conn.execute(
        "SELECT content_hash FROM beliefs WHERE id = ?",
        (phantom_id,),
    ).fetchone()
    assert got["content_hash"] == v1_hash
    assert v1_hash != v2_hash

    s._conn.execute(
        "DELETE FROM schema_meta WHERE key = ?",
        (SCHEMA_META_SPECULATIVE_HASH_V2_COMPLETE,),
    )
    s._conn.commit()
    s.close()

    s2 = MemoryStore(db)
    try:
        row = s2._conn.execute(
            "SELECT content_hash FROM beliefs WHERE id = ?",
            (phantom_id,),
        ).fetchone()
        assert row["content_hash"] == v2_hash, (
            f"migration did not rewrite hash from v1 to v2; "
            f"got {row['content_hash']!r}"
        )
        assert s2.get_schema_meta(SCHEMA_META_SPECULATIVE_HASH_V2_COMPLETE)
    finally:
        s2.close()


def test_migration_is_idempotent_on_second_open(tmp_path) -> None:
    """Marker present → second open is a no-op (no UPDATEs issued)."""
    db = str(tmp_path / "spec_idem.db")
    s = MemoryStore(db)
    phantom_id, v1_hash, _v2 = _seed_v1_shaped_phantom(s)
    # First open already stamped the marker; the seed rewrote the row's
    # hash to v1 AFTER that. Running the migration again with the marker
    # in place must short-circuit — the row stays at v1.
    rewritten = s._maybe_rehash_speculative_v2()
    assert rewritten == 0
    row = s._conn.execute(
        "SELECT content_hash FROM beliefs WHERE id = ?",
        (phantom_id,),
    ).fetchone()
    assert row["content_hash"] == v1_hash
    s.close()


def test_migration_skips_speculative_row_without_audit_trail(tmp_path) -> None:
    """A speculative row with no wonder_ingest corroboration → skipped.

    Generator cannot be recovered, so the migration leaves the row's
    hash unchanged rather than guessing.
    """
    db = str(tmp_path / "spec_ghost.db")
    s = MemoryStore(db)
    s.insert_belief(_constituent("a"))
    s.insert_belief(_constituent("b"))
    # Insert a speculative belief WITHOUT going through wonder_ingest —
    # no corroboration row will exist.
    ghost = Belief(
        id="ghost",
        content="speculative without audit",
        content_hash="ghost-hash-value",
        alpha=0.3,
        beta=1.0,
        type=BELIEF_SPECULATIVE,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-01T00:00:00+00:00",
        last_retrieved_at=None,
        origin=ORIGIN_SPECULATIVE,
        retention_class="snapshot",
    )
    s.insert_belief(ghost)
    # Drop the marker so the next call to the migration actually runs.
    s._conn.execute(
        "DELETE FROM schema_meta WHERE key = ?",
        (SCHEMA_META_SPECULATIVE_HASH_V2_COMPLETE,),
    )
    s._conn.commit()
    rewritten = s._maybe_rehash_speculative_v2()
    assert rewritten == 0
    row = s._conn.execute(
        "SELECT content_hash FROM beliefs WHERE id = ?",
        ("ghost",),
    ).fetchone()
    assert row["content_hash"] == "ghost-hash-value"
    assert s.get_schema_meta(SCHEMA_META_SPECULATIVE_HASH_V2_COMPLETE)
    s.close()


def test_migration_marker_stamped_on_fresh_store(tmp_path) -> None:
    """A fresh store with no speculative rows still stamps the marker
    so subsequent opens short-circuit. Same idempotency shape as the
    other one-shot backfills."""
    db = str(tmp_path / "fresh.db")
    s = MemoryStore(db)
    assert s.get_schema_meta(SCHEMA_META_SPECULATIVE_HASH_V2_COMPLETE)
    s.close()
