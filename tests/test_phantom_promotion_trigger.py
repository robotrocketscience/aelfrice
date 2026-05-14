"""End-to-end tests for the phantom promotion trigger (#550).

Six test groups:
  (a) Surface A: explicit promote() on a phantom belief.
  (b) Surface B: exact content_hash match via find_phantom_lock_matches().
  (c) Surface B: Jaccard ≥ 0.9 match.
  (d) Surface B: no match — lock on unrelated text leaves phantoms alone.
  (e) Idempotency on both surfaces.
  (f) audit_log row contents per trigger.

Integration tests for the CLI path use _cmd_lock directly, following
the pattern in test_cli_lock_via_worker.py.
"""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SPECULATIVE,
    LOCK_NONE,
    ORIGIN_SPECULATIVE,
    ORIGIN_USER_VALIDATED,
    RETENTION_FACT,
    RETENTION_SNAPSHOT,
    Belief,
    Phantom,
)
from aelfrice.promotion import (
    SOURCE_PROMOTE_PHANTOM_LOCK_MATCH,
    SOURCE_PROMOTE_USER_VALIDATED,
    find_phantom_lock_matches,
    promote,
)
from aelfrice.store import MemoryStore
from aelfrice.wonder.lifecycle import wonder_ingest


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _belief(bid: str, *, content: str, content_hash: str,
            origin: str = ORIGIN_SPECULATIVE,
            btype: str = BELIEF_SPECULATIVE,
            retention: str = RETENTION_SNAPSHOT) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=content_hash,
        alpha=0.3,
        beta=1.0,
        type=btype,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-05-10T00:00:00Z",
        last_retrieved_at=None,
        origin=origin,
        retention_class=retention,
    )


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed_phantom(
    store: MemoryStore,
    bid: str,
    content: str,
) -> Belief:
    """Insert a speculative phantom with content_hash = sha256(content)."""
    b = _belief(bid, content=content, content_hash=_content_hash(content))
    store.insert_belief(b)
    return b


def _seed_store() -> MemoryStore:
    return MemoryStore(":memory:")


# ---------------------------------------------------------------------------
# (a) Surface A: explicit promote() on a phantom
# ---------------------------------------------------------------------------


def test_surface_a_promotes_speculative_to_user_validated() -> None:
    """Hypothesis: explicit promote() on a phantom flips origin."""
    store = _seed_store()
    _seed_phantom(store, "P1", "the system defaults to deterministic retrieval")
    result = promote(store, "P1")
    assert result.prior_origin == ORIGIN_SPECULATIVE
    assert result.new_origin == ORIGIN_USER_VALIDATED
    after = store.get_belief("P1")
    assert after is not None
    assert after.origin == ORIGIN_USER_VALIDATED


def test_surface_a_writes_user_validated_audit_row() -> None:
    """Hypothesis: Surface A audit row carries promotion:user_validated."""
    store = _seed_store()
    _seed_phantom(store, "P1", "deterministic retrieval is preferred")
    promote(store, "P1")
    ev = store.list_feedback_events()[0]
    assert ev.belief_id == "P1"
    assert ev.source == SOURCE_PROMOTE_USER_VALIDATED
    assert ev.valence == 0.0


# ---------------------------------------------------------------------------
# (b) Surface B: exact content_hash match
# ---------------------------------------------------------------------------


def test_surface_b_exact_hash_match_returns_id() -> None:
    """Hypothesis: a phantom whose content_hash == sha256(lock_text) matches."""
    store = _seed_store()
    lock_text = "atomic commits beat batched"
    _seed_phantom(store, "P2", lock_text)  # content_hash = sha256(lock_text)

    matched = find_phantom_lock_matches(store, lock_text)
    assert matched == ["P2"]


def test_surface_b_exact_hash_match_does_not_match_different_text() -> None:
    """Hypothesis: a phantom with different content does not match by hash."""
    store = _seed_store()
    _seed_phantom(store, "P3", "something completely unrelated to the lock text")
    matched = find_phantom_lock_matches(store, "atomic commits beat batched")
    assert matched == []


def test_surface_b_exact_hash_promotes_via_cmd_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: _cmd_lock auto-promotes a phantom matched by content_hash."""
    from aelfrice.cli import _cmd_lock

    db = tmp_path / "exact-hash.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    store = MemoryStore(str(db))

    lock_text = "atomic commits beat batched"
    _seed_phantom(store, "PHANTOM-EXACT", lock_text)
    store.close()

    # Re-open via monkeypatched env (same path).
    out = io.StringIO()
    rc = _cmd_lock(
        argparse.Namespace(statement=lock_text, session_id=None, doc_uri=None),
        out,
    )
    assert rc == 0

    store2 = MemoryStore(str(db))
    promoted = store2.get_belief("PHANTOM-EXACT")
    assert promoted is not None
    assert promoted.origin == ORIGIN_USER_VALIDATED
    store2.close()

    output = out.getvalue()
    assert "promoted phantom: PHANTOM-EXACT" in output


# ---------------------------------------------------------------------------
# (c) Surface B: Jaccard ≥ 0.9 match
# ---------------------------------------------------------------------------


def test_surface_b_jaccard_match_returns_id() -> None:
    """Hypothesis: a phantom with Jaccard ≥ 0.9 against lock_text matches."""
    store = _seed_store()
    # "atomic commits are better than batching" vs "atomic commits beat batched"
    # Both share tokens: atomic, commits — let's make a clear ≥ 0.9 pair.
    lock_text = "deterministic retrieval is the default strategy"
    # Phantom text: only minor stopword difference
    phantom_text = "deterministic retrieval default strategy"
    _seed_phantom(store, "P4", phantom_text)
    matched = find_phantom_lock_matches(store, lock_text)
    assert "P4" in matched


def test_surface_b_jaccard_below_threshold_no_match() -> None:
    """Hypothesis: Jaccard < 0.9 does not trigger a match."""
    store = _seed_store()
    # Clearly below-threshold: only one shared content word out of many.
    _seed_phantom(store, "P5", "retrieval uses a completely different strategy here")
    matched = find_phantom_lock_matches(store, "atomic commits beat batched")
    assert "P5" not in matched


def test_surface_b_jaccard_promotes_via_cmd_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: _cmd_lock auto-promotes a Jaccard-matched phantom."""
    from aelfrice.cli import _cmd_lock

    db = tmp_path / "jaccard.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    store = MemoryStore(str(db))

    lock_text = "deterministic retrieval is the default strategy"
    phantom_text = "deterministic retrieval default strategy"
    _seed_phantom(store, "PHANTOM-JACCARD", phantom_text)
    store.close()

    out = io.StringIO()
    rc = _cmd_lock(
        argparse.Namespace(statement=lock_text, session_id=None, doc_uri=None),
        out,
    )
    assert rc == 0

    store2 = MemoryStore(str(db))
    promoted = store2.get_belief("PHANTOM-JACCARD")
    assert promoted is not None
    assert promoted.origin == ORIGIN_USER_VALIDATED
    store2.close()

    assert "promoted phantom: PHANTOM-JACCARD" in out.getvalue()


# ---------------------------------------------------------------------------
# (d) Surface B: no match — unrelated text leaves phantoms alone
# ---------------------------------------------------------------------------


def test_surface_b_unrelated_lock_leaves_phantom_unpromoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: locking unrelated text does not promote phantoms."""
    from aelfrice.cli import _cmd_lock

    db = tmp_path / "no-match.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    store = MemoryStore(str(db))
    _seed_phantom(store, "PHANTOM-QUIET", "deterministic retrieval default strategy")
    store.close()

    out = io.StringIO()
    rc = _cmd_lock(
        argparse.Namespace(
            statement="completely unrelated belief about coffee preferences",
            session_id=None, doc_uri=None,
        ),
        out,
    )
    assert rc == 0

    store2 = MemoryStore(str(db))
    untouched = store2.get_belief("PHANTOM-QUIET")
    assert untouched is not None
    assert untouched.origin == ORIGIN_SPECULATIVE
    store2.close()

    assert "promoted phantom" not in out.getvalue()


def test_find_phantom_lock_matches_returns_empty_on_no_phantoms() -> None:
    """Hypothesis: scanner returns [] when no speculative beliefs exist."""
    store = _seed_store()
    # Only a regular belief — no phantoms.
    store.insert_belief(_belief(
        "REG", content="regular belief",
        content_hash=_content_hash("regular belief"),
        origin="agent_inferred", btype=BELIEF_FACTUAL,
        retention=RETENTION_FACT,
    ))
    matched = find_phantom_lock_matches(store, "regular belief")
    assert matched == []


# ---------------------------------------------------------------------------
# (e) Idempotency on both surfaces
# ---------------------------------------------------------------------------


def test_surface_a_idempotent_no_double_audit() -> None:
    """Hypothesis: Surface A re-promotion is a no-op; one audit row only."""
    store = _seed_store()
    _seed_phantom(store, "P6", "idempotency test content")
    first = promote(store, "P6")
    second = promote(store, "P6")
    assert first.audit_event_id is not None
    assert second.audit_event_id is None
    assert second.already_validated is True
    assert store.count_feedback_events() == 1


def test_surface_b_idempotent_via_find_then_promote() -> None:
    """Hypothesis: promoting an already-promoted phantom via find+promote is a no-op."""
    store = _seed_store()
    lock_text = "idempotency via lock match"
    _seed_phantom(store, "P7", lock_text)
    # First promotion
    ids = find_phantom_lock_matches(store, lock_text)
    assert ids == ["P7"]
    first = promote(store, "P7", source_label=SOURCE_PROMOTE_PHANTOM_LOCK_MATCH)
    assert first.already_validated is False
    assert first.audit_event_id is not None
    # Already promoted — find still returns it, but promote is a no-op.
    ids2 = find_phantom_lock_matches(store, lock_text)
    assert ids2 == []  # promoted belief is no longer speculative; excluded
    assert store.count_feedback_events() == 1


def test_surface_b_cmd_lock_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hypothesis: running _cmd_lock twice on the same text promotes once."""
    from aelfrice.cli import _cmd_lock

    db = tmp_path / "idempotent.db"
    monkeypatch.setenv("AELFRICE_DB", str(db))
    store = MemoryStore(str(db))
    lock_text = "idempotency via lock match"
    _seed_phantom(store, "PHANTOM-IDEM", lock_text)
    store.close()

    _cmd_lock(
        argparse.Namespace(statement=lock_text, session_id=None, doc_uri=None),
        io.StringIO(),
    )
    _cmd_lock(
        argparse.Namespace(statement=lock_text, session_id=None, doc_uri=None),
        io.StringIO(),
    )

    store2 = MemoryStore(str(db))
    events = [e for e in store2.list_feedback_events()
              if e.source == SOURCE_PROMOTE_PHANTOM_LOCK_MATCH]
    assert len(events) == 1  # promoted exactly once
    store2.close()


# ---------------------------------------------------------------------------
# (f) Audit-log row contents per trigger
# ---------------------------------------------------------------------------


def test_surface_a_audit_row_shape() -> None:
    """Hypothesis: Surface A audit row has belief_id, valence=0.0, correct source."""
    store = _seed_store()
    _seed_phantom(store, "PA", "audit row test for surface a")
    promote(store, "PA", now="2026-05-10T12:00:00Z")
    ev = store.list_feedback_events()[0]
    assert ev.belief_id == "PA"
    assert ev.valence == 0.0
    assert ev.source == SOURCE_PROMOTE_USER_VALIDATED
    assert ev.created_at == "2026-05-10T12:00:00Z"


def test_surface_b_audit_row_shape() -> None:
    """Hypothesis: Surface B audit row carries phantom_lock_match source."""
    store = _seed_store()
    lock_text = "surface b audit row test"
    _seed_phantom(store, "PB", lock_text)
    ids = find_phantom_lock_matches(store, lock_text)
    assert ids == ["PB"]
    promote(
        store, "PB",
        source_label=SOURCE_PROMOTE_PHANTOM_LOCK_MATCH,
        now="2026-05-10T13:00:00Z",
    )
    ev = store.list_feedback_events()[0]
    assert ev.belief_id == "PB"
    assert ev.source == SOURCE_PROMOTE_PHANTOM_LOCK_MATCH
    assert ev.source == "promotion:phantom_lock_match"
    assert ev.valence == 0.0
    assert ev.created_at == "2026-05-10T13:00:00Z"


def test_surface_b_audit_row_distinguishable_from_surface_a() -> None:
    """Hypothesis: Surface A and B audit rows differ in source label."""
    assert SOURCE_PROMOTE_USER_VALIDATED != SOURCE_PROMOTE_PHANTOM_LOCK_MATCH
    assert SOURCE_PROMOTE_PHANTOM_LOCK_MATCH == "promotion:phantom_lock_match"
    assert SOURCE_PROMOTE_USER_VALIDATED == "promotion:user_validated"


# ---------------------------------------------------------------------------
# (g) Wonder-ingest integration: promote a wonder_ingest phantom
# ---------------------------------------------------------------------------


def test_wonder_ingest_phantom_promotes_via_surface_a() -> None:
    """Hypothesis: a phantom written by wonder_ingest promotes via Surface A."""
    store = _seed_store()
    # Insert constituents first.
    store.insert_belief(_belief(
        "C1", content="belief alpha", content_hash=_content_hash("belief alpha"),
        origin="agent_inferred", btype=BELIEF_FACTUAL, retention=RETENTION_FACT,
    ))
    store.insert_belief(_belief(
        "C2", content="belief beta", content_hash=_content_hash("belief beta"),
        origin="agent_inferred", btype=BELIEF_FACTUAL, retention=RETENTION_FACT,
    ))
    phantom = Phantom(
        constituent_belief_ids=("C1", "C2"),
        generator="test_generator",
        content="alpha and beta together imply gamma",
        score=0.8,
    )
    result = wonder_ingest(store, [phantom])
    assert result.inserted == 1

    # Find the inserted phantom id.
    active = store.list_active_speculative_beliefs()
    assert len(active) == 1
    phantom_id = active[0].id

    pr = promote(store, phantom_id)
    assert pr.prior_origin == ORIGIN_SPECULATIVE
    assert pr.new_origin == ORIGIN_USER_VALIDATED
