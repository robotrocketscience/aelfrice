"""CRUD + FTS5 tests for the SQLite store."""
from __future__ import annotations

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    BELIEF_SCOPE_GLOBAL,
    EDGE_SUPPORTS,
    INGEST_SOURCE_CLI_REMEMBER,
    INGEST_SOURCE_LEGACY_UNKNOWN,
    LOCK_NONE,
    RETENTION_FACT,
    RETENTION_TRANSIENT,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def _mk_belief(
    bid: str = "b1",
    content: str = "the sky is blue",
    alpha: float = 1.0,
    beta: float = 1.0,
    demotion_pressure: int = 0,
) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash="h_" + bid,
        alpha=alpha,
        beta=beta,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=demotion_pressure,
        created_at="2026-04-25T00:00:00Z",
        last_retrieved_at=None,
    )


def test_belief_insert_and_get_roundtrip() -> None:
    s = MemoryStore(":memory:")
    b = _mk_belief()
    s.insert_belief(b)
    got = s.get_belief("b1")
    assert got is not None
    assert got.id == b.id
    assert got.content == b.content
    assert got.content_hash == b.content_hash
    assert got.alpha == b.alpha
    assert got.beta == b.beta
    assert got.type == b.type
    assert got.lock_level == b.lock_level
    assert got.locked_at == b.locked_at
    assert got.demotion_pressure == b.demotion_pressure
    assert got.created_at == b.created_at
    assert got.last_retrieved_at == b.last_retrieved_at


def test_belief_update_persists_alpha_beta_and_demotion() -> None:
    s = MemoryStore(":memory:")
    b = _mk_belief()
    s.insert_belief(b)
    b.alpha = 5.5
    b.beta = 2.25
    b.demotion_pressure = 4
    s.update_belief(b)
    got = s.get_belief("b1")
    assert got is not None
    assert got.alpha == 5.5
    assert got.beta == 2.25
    assert got.demotion_pressure == 4


def test_belief_update_persists_full_row_columns() -> None:
    """update_belief docstring promises full-row semantics: every Belief
    dataclass field that maps to a `beliefs` column must round-trip. This
    test mutates each column previously absent from the SET clause
    (#708) and asserts the on-disk row reflects the mutation."""
    s = MemoryStore(":memory:")
    b = _mk_belief()
    s.insert_belief(b)
    b.hibernation_score = 0.42
    b.activation_condition = '{"on": "next_retrieval"}'
    b.retention_class = RETENTION_FACT
    b.valid_to = "2026-12-31T23:59:59Z"
    b.scope = BELIEF_SCOPE_GLOBAL
    s.update_belief(b)
    got = s.get_belief("b1")
    assert got is not None
    assert got.hibernation_score == 0.42
    assert got.activation_condition == '{"on": "next_retrieval"}'
    assert got.retention_class == RETENTION_FACT
    assert got.valid_to == "2026-12-31T23:59:59Z"
    assert got.scope == BELIEF_SCOPE_GLOBAL


def test_belief_update_rejects_invalid_retention_class() -> None:
    """update_belief validates retention_class on the same allowlist as
    insert_belief; an invalid value raises ValueError before any UPDATE."""
    s = MemoryStore(":memory:")
    b = _mk_belief()
    b.retention_class = RETENTION_TRANSIENT
    s.insert_belief(b)
    b.retention_class = "bogus"
    with pytest.raises(ValueError, match="invalid retention_class"):
        s.update_belief(b)
    # On-disk row unchanged because validation fired before UPDATE.
    got = s.get_belief("b1")
    assert got is not None
    assert got.retention_class == RETENTION_TRANSIENT


def test_belief_delete_returns_none() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief())
    s.delete_belief("b1")
    assert s.get_belief("b1") is None


def test_edge_insert_get_update_delete() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("a"))
    s.insert_belief(_mk_belief("b", content="thing two"))
    e = Edge(src="a", dst="b", type=EDGE_SUPPORTS, weight=0.7)
    s.insert_edge(e)
    got = s.get_edge("a", "b", EDGE_SUPPORTS)
    assert got is not None
    assert got.weight == 0.7

    e.weight = 0.9
    s.update_edge(e)
    got2 = s.get_edge("a", "b", EDGE_SUPPORTS)
    assert got2 is not None
    assert got2.weight == 0.9

    s.delete_edge("a", "b", EDGE_SUPPORTS)
    assert s.get_edge("a", "b", EDGE_SUPPORTS) is None


def test_fts5_search_finds_belief_by_keyword() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1", content="apples are red and crunchy"))
    s.insert_belief(_mk_belief("b2", content="bananas are yellow"))
    s.insert_belief(_mk_belief("b3", content="grapes are purple"))

    hits = s.search_beliefs("bananas")
    assert len(hits) == 1
    assert hits[0].id == "b2"

    hits2 = s.search_beliefs("are")
    ids = {h.id for h in hits2}
    assert ids == {"b1", "b2", "b3"}


def test_fts5_search_after_update_reflects_new_content() -> None:
    s = MemoryStore(":memory:")
    b = _mk_belief("b1", content="initial wording about cats")
    s.insert_belief(b)
    assert {h.id for h in s.search_beliefs("cats")} == {"b1"}

    b.content = "rewritten to mention dogs only"
    b.content_hash = "h_b1_v2"
    s.update_belief(b)
    assert s.search_beliefs("cats") == []
    assert {h.id for h in s.search_beliefs("dogs")} == {"b1"}


# --- stamp_retrieved (issue #222) ----------------------------------------


def test_stamp_retrieved_populates_column() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1"))
    s.insert_belief(_mk_belief("b2"))
    n = s.stamp_retrieved(["b1", "b2"])
    assert n == 2
    assert s.get_belief("b1").last_retrieved_at is not None  # type: ignore[union-attr]
    assert s.get_belief("b2").last_retrieved_at is not None  # type: ignore[union-attr]


def test_stamp_retrieved_uses_explicit_ts_when_given() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1"))
    s.stamp_retrieved(["b1"], ts="2026-04-28T12:00:00Z")
    assert s.get_belief("b1").last_retrieved_at == "2026-04-28T12:00:00Z"  # type: ignore[union-attr]


def test_stamp_retrieved_empty_input_is_noop() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1"))
    assert s.stamp_retrieved([]) == 0
    assert s.get_belief("b1").last_retrieved_at is None  # type: ignore[union-attr]


def test_stamp_retrieved_silently_skips_missing_ids() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1"))
    n = s.stamp_retrieved(["b1", "ghost"])
    assert n == 1
    assert s.get_belief("b1").last_retrieved_at is not None  # type: ignore[union-attr]


def test_stamp_retrieved_overwrites_prior_timestamp() -> None:
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1"))
    s.stamp_retrieved(["b1"], ts="2026-04-28T01:00:00Z")
    s.stamp_retrieved(["b1"], ts="2026-04-28T02:00:00Z")
    assert s.get_belief("b1").last_retrieved_at == "2026-04-28T02:00:00Z"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# list_canonical_orphans (#725)
# ---------------------------------------------------------------------------

_TS = "2026-05-12T00:00:00+00:00"


def test_list_canonical_orphans_empty_store() -> None:
    """Empty store → empty list."""
    s = MemoryStore(":memory:")
    assert s.list_canonical_orphans() == []


def test_list_canonical_orphans_no_log_rows() -> None:
    """Belief with zero ingest_log rows is a canonical orphan (pre-#205 case)."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1"))
    result = s.list_canonical_orphans()
    assert result == [("b1", "h_b1")]


def test_list_canonical_orphans_all_legacy_rows() -> None:
    """Belief whose only log rows are legacy_unknown → canonical orphan."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1"))
    s.record_ingest(
        source_kind=INGEST_SOURCE_LEGACY_UNKNOWN,
        source_path=None,
        raw_text="content b1",
        derived_belief_ids=["b1"],
        ts=_TS,
    )
    result = s.list_canonical_orphans()
    assert result == [("b1", "h_b1")]


def test_list_canonical_orphans_non_legacy_row_excludes_belief() -> None:
    """Belief with a non-legacy log row is not an orphan."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1"))
    s.record_ingest(
        source_kind=INGEST_SOURCE_CLI_REMEMBER,
        source_path=None,
        raw_text="content b1",
        derived_belief_ids=["b1"],
        ts=_TS,
    )
    assert s.list_canonical_orphans() == []


def test_list_canonical_orphans_mixed_beliefs() -> None:
    """Only all-legacy beliefs surface; non-legacy-covered beliefs are excluded."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1"))  # will be orphan
    s.insert_belief(_mk_belief("b2"))  # has a real log row → not orphan
    s.insert_belief(_mk_belief("b3"))  # no log rows at all → orphan
    s.record_ingest(
        source_kind=INGEST_SOURCE_LEGACY_UNKNOWN,
        source_path=None,
        raw_text="content b1",
        derived_belief_ids=["b1"],
        ts=_TS,
    )
    s.record_ingest(
        source_kind=INGEST_SOURCE_CLI_REMEMBER,
        source_path=None,
        raw_text="content b2",
        derived_belief_ids=["b2"],
        ts=_TS,
    )
    result = s.list_canonical_orphans()
    assert result == [("b1", "h_b1"), ("b3", "h_b3")]


def test_list_canonical_orphans_mixed_log_rows_same_belief() -> None:
    """A belief with both a legacy and a non-legacy row is NOT an orphan."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("b1"))
    s.record_ingest(
        source_kind=INGEST_SOURCE_LEGACY_UNKNOWN,
        source_path=None,
        raw_text="content b1",
        derived_belief_ids=["b1"],
        ts=_TS,
    )
    s.record_ingest(
        source_kind=INGEST_SOURCE_CLI_REMEMBER,
        source_path=None,
        raw_text="content b1",
        derived_belief_ids=["b1"],
        ts=_TS,
    )
    assert s.list_canonical_orphans() == []


def test_list_canonical_orphans_limit() -> None:
    """limit= caps the returned list."""
    s = MemoryStore(":memory:")
    for i in range(5):
        s.insert_belief(_mk_belief(f"b{i}"))
    result = s.list_canonical_orphans(limit=3)
    assert len(result) == 3


def test_list_canonical_orphans_order_by_id_asc() -> None:
    """Results are returned in id ASC order (same as list_belief_ids)."""
    s = MemoryStore(":memory:")
    for bid in ["bz", "ba", "bm"]:
        s.insert_belief(_mk_belief(bid))
    ids_orphans = [r[0] for r in s.list_canonical_orphans()]
    ids_list = s.list_belief_ids()
    assert ids_orphans == sorted(ids_orphans)
    assert ids_orphans == ids_list


def test_list_canonical_orphans_no_log_rows_returns_content_hash() -> None:
    """Tuples carry the correct content_hash from the beliefs row."""
    s = MemoryStore(":memory:")
    s.insert_belief(_mk_belief("bx"))
    result = s.list_canonical_orphans()
    assert len(result) == 1
    bid, ch = result[0]
    assert bid == "bx"
    assert ch == "h_bx"
