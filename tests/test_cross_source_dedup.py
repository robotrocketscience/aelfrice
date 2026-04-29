"""Cross-source dedup via content_hash pre-check (#254).

belief_id is sha256(source‖text); content_hash is sha256(text). The
same sentence ingested under two different sources used to land as
two separate rows because the id-based dedup check missed it. The
ingest path now consults `get_belief_by_content_hash` after the
id-miss and records a corroboration against the canonical row
instead of inserting a parallel one. See issue #254.
"""
from __future__ import annotations

from pathlib import Path

from aelfrice.ingest import ingest_turn
from aelfrice.store import MemoryStore


def _all_beliefs(store: MemoryStore) -> list:
    return [store.get_belief(bid) for bid in store.list_belief_ids()]


def test_cross_source_reingest_does_not_inflate_rows(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "t.db"))
    try:
        text = "The default port is 8080 for the service."
        n1 = ingest_turn(store, text, source="source_a")
        assert n1 == 1
        n2 = ingest_turn(store, text, source="source_b")
        # Same content_hash → no new belief row.
        assert n2 == 0
        beliefs = _all_beliefs(store)
        hashes = {b.content_hash for b in beliefs if b is not None}
        assert len(hashes) == 1
        assert len(beliefs) == 1
    finally:
        store.close()


def test_cross_source_reingest_records_corroboration_against_canonical(
    tmp_path: Path,
) -> None:
    store = MemoryStore(str(tmp_path / "t.db"))
    try:
        text = "The configuration directory is /etc/aelfrice."
        ingest_turn(store, text, source="source_a")
        beliefs = _all_beliefs(store)
        assert len(beliefs) == 1
        canonical_id = beliefs[0].id

        ingest_turn(store, text, source="source_b")
        ingest_turn(store, text, source="source_c")

        # Two corroborations on the canonical row, no new rows.
        assert store.count_corroborations(canonical_id) == 2
        assert len(_all_beliefs(store)) == 1
    finally:
        store.close()


def test_cross_source_canonical_id_is_first_writer(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "t.db"))
    try:
        text = "Cache TTL is 60 seconds by default."
        ingest_turn(store, text, source="alpha")
        first = _all_beliefs(store)[0]
        ingest_turn(store, text, source="beta")
        # Canonical id stable: still the alpha-source id.
        beliefs = _all_beliefs(store)
        assert len(beliefs) == 1
        assert beliefs[0].id == first.id
    finally:
        store.close()


def test_bulk_cross_source_skips_corroboration(tmp_path: Path) -> None:
    """bulk=True suppresses corroboration on the cross-source path,
    matching the same-source bulk semantics."""
    store = MemoryStore(str(tmp_path / "t.db"))
    try:
        text = "Retries are capped at 3 by default."
        ingest_turn(store, text, source="src_a")
        canonical_id = _all_beliefs(store)[0].id
        ingest_turn(store, text, source="src_b", bulk=True)
        assert store.count_corroborations(canonical_id) == 0
        assert len(_all_beliefs(store)) == 1
    finally:
        store.close()
