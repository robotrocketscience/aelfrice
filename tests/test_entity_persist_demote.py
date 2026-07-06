"""#1096 entity-persistence demotion lane.

Covers the store S1 computation, the log-additive penalty, the flag
resolver, and the retrieve_v2 / _l1_hits integration (default-off
byte-identical; on = reorder-not-drop; short-circuit force). Synthetic
fixtures only.
"""
from __future__ import annotations

import math

import pytest

from aelfrice.retrieval import (
    ENTITY_PERSIST_DEMOTE_EPS,
    ENTITY_PERSIST_DEMOTE_WEIGHT,
    _entity_persist_penalty,
    _l1_hits,
    is_entity_persist_demote_enabled,
    retrieve_v2,
)
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief
from aelfrice.store import MemoryStore


def _mk(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-06-01T00:00:00Z",
        last_retrieved_at=None,
    )


def _add_entity(store: MemoryStore, bid: str, lower: str, kind: str) -> None:
    store._conn.execute(
        "INSERT INTO belief_entities(belief_id, entity_lower, entity_raw, "
        "kind, span_start, span_end) VALUES (?,?,?,?,0,0)",
        (bid, lower, lower, kind),
    )
    store._conn.commit()


# --- entity_persistence_scores -------------------------------------------


def test_s1_durable_only() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("b1", "plain text no entities here"))
        _add_entity(s, "b1", "src/foo.py", "file_path")
        out = s.entity_persistence_scores(["b1"])
    finally:
        s.close()
    assert out["b1"] == pytest.approx(1 / (1 + 0 + 1))  # 0.5


def test_s1_transient_only_bare_number() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("b1", "plain text no entities here"))
        _add_entity(s, "b1", "#879", "identifier")  # bare number -> transient
        out = s.entity_persistence_scores(["b1"])
    finally:
        s.close()
    assert out["b1"] == pytest.approx(0 / (0 + 1 + 1))  # 0.0


def test_s1_symbol_identifier_is_durable() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("b1", "plain text no entities here"))
        _add_entity(s, "b1", "retrieve_v2", "identifier")  # has letters
        out = s.entity_persistence_scores(["b1"])
    finally:
        s.close()
    assert out["b1"] == pytest.approx(0.5)


def test_s1_mixed() -> None:
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("b1", "plain text no entities here"))
        _add_entity(s, "b1", "src/a.py", "file_path")   # durable
        _add_entity(s, "b1", "#12", "identifier")        # transient
        _add_entity(s, "b1", "v3.1", "version")          # transient
        out = s.entity_persistence_scores(["b1"])
    finally:
        s.close()
    assert out["b1"] == pytest.approx(1 / (1 + 2 + 1))  # 0.25


def test_s1_entity_free_belief_absent() -> None:
    """A belief with no entity rows is absent from the map — the caller
    applies no demotion, so entity-free durable content is never hit."""
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("plain", "no extractable entities at all"))
        # ensure the extractor didn't add rows for this content
        s._conn.execute("DELETE FROM belief_entities WHERE belief_id='plain'")
        s._conn.commit()
        out = s.entity_persistence_scores(["plain"])
    finally:
        s.close()
    assert "plain" not in out


def test_s1_empty_input_no_sql() -> None:
    s = MemoryStore(":memory:")
    try:
        assert s.entity_persistence_scores([]) == {}
    finally:
        s.close()


# --- penalty --------------------------------------------------------------


def test_penalty_off_when_ep_none() -> None:
    assert _entity_persist_penalty(None, "b1") == 0.0


def test_penalty_zero_for_absent_belief() -> None:
    assert _entity_persist_penalty({"other": 0.5}, "b1") == 0.0


def test_penalty_value() -> None:
    got = _entity_persist_penalty({"b1": 0.5}, "b1")
    assert got == pytest.approx(
        ENTITY_PERSIST_DEMOTE_WEIGHT * math.log(0.5 + ENTITY_PERSIST_DEMOTE_EPS)
    )


def test_penalty_monotone_low_s1_more_negative() -> None:
    lo = _entity_persist_penalty({"b": 0.0}, "b")
    hi = _entity_persist_penalty({"b": 1.0}, "b")
    assert lo < hi
    assert hi == 0.0  # clamped: a well-grounded belief is never boosted
    assert lo < 0.0   # durable-free is demoted


# --- resolver -------------------------------------------------------------


def test_resolver_default_off() -> None:
    assert is_entity_persist_demote_enabled(None) is False


def test_resolver_kwarg_true() -> None:
    assert is_entity_persist_demote_enabled(True) is True


def test_resolver_env_overrides_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AELFRICE_ENTITY_PERSIST_DEMOTE", "0")
    assert is_entity_persist_demote_enabled(True) is False
    monkeypatch.setenv("AELFRICE_ENTITY_PERSIST_DEMOTE", "1")
    assert is_entity_persist_demote_enabled(False) is True


# --- integration: retrieve_v2 + _l1_hits ----------------------------------


def _seed_two(store: MemoryStore) -> None:
    # Both match the query token; durable grounds to a file, ephemeral to
    # a bare PR number. Same base posterior.
    store.insert_belief(_mk("durable", "widget the module lives here"))
    store.insert_belief(_mk("ephemeral", "widget rebased and pushed"))
    for bid in ("durable", "ephemeral"):
        store._conn.execute(
            "DELETE FROM belief_entities WHERE belief_id=?", (bid,)
        )
    _add_entity(store, "durable", "src/widget.py", "file_path")
    _add_entity(store, "ephemeral", "#412", "identifier")
    store._conn.commit()


def test_retrieve_v2_off_is_byte_identical_default() -> None:
    s = MemoryStore(":memory:")
    try:
        _seed_two(s)
        base = [b.id for b in retrieve_v2(s, "widget").beliefs]
        off = [
            b.id
            for b in retrieve_v2(s, "widget", use_entity_persist_demote=False).beliefs
        ]
    finally:
        s.close()
    assert base == off  # default == explicit-off


def test_retrieve_v2_on_reorders_not_drops() -> None:
    s = MemoryStore(":memory:")
    try:
        _seed_two(s)
        off = [b.id for b in retrieve_v2(s, "widget", use_entity_persist_demote=False).beliefs]
        on = [b.id for b in retrieve_v2(s, "widget", use_entity_persist_demote=True).beliefs]
    finally:
        s.close()
    assert set(off) == set(on)  # reorder, never drop
    # ephemeral (bare-#) must not rank above durable under the demotion
    assert on.index("durable") <= on.index("ephemeral")


def test_l1_hits_short_circuit_forced_when_flag_on() -> None:
    """posterior_weight=0.0 short-circuits to the byte-identical FTS
    contract; the flag must force the rerank loop (like gamma/zeta)."""
    s = MemoryStore(":memory:")
    try:
        _seed_two(s)
        forced = _l1_hits(
            s, "widget", l1_limit=10, posterior_weight=0.0,
            use_entity_persist_demote=True,
        )
        # The demotion ran: durable is not below ephemeral.
        ids = [b.id for b in forced]
        assert set(ids) == {"durable", "ephemeral"}
        assert ids.index("durable") <= ids.index("ephemeral")
    finally:
        s.close()
