"""Exploration-slot store surface (#1176 proposal 5).

`exploration_pool` is a filter, so every clause gets a test that fails if
that clause alone is dropped, plus a control proving the pool is not simply
empty. `record_exploration` is an audit row, so the tests are about what
survives a round trip.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore, hash_query


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(str(tmp_path / "memory.db"))
    yield s
    s.close()


def _mk(bid: str, content: str, *, lock: str = LOCK_NONE) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock,
        locked_at="2026-07-31T00:00:00Z" if lock == LOCK_USER else None,
        created_at="2026-07-31T00:00:00Z",
        last_retrieved_at=None,
    )


def _seed(store: MemoryStore, n: int = 4) -> list[str]:
    """n beliefs that all match the query `bananas`."""
    ids = []
    for i in range(n):
        b = _mk(f"B{i}", f"bananas ripen in the kitchen number {i}")
        store.insert_belief(b)
        ids.append(b.id)
    return ids


# --- the pool filter -------------------------------------------------------


def test_unexplored_matching_beliefs_are_in_the_pool(store: MemoryStore) -> None:
    """Control. Every exclusion test below would pass on an always-empty pool."""
    ids = _seed(store)
    assert sorted(store.exploration_pool("bananas")) == sorted(ids)


def test_non_matching_beliefs_are_excluded(store: MemoryStore) -> None:
    """The pool is FTS-restricted so a slot is never off-topic noise."""
    _seed(store)
    store.insert_belief(_mk("OTHER", "the harbour ferry leaves at noon"))
    assert "OTHER" not in store.exploration_pool("bananas")


def test_locked_beliefs_are_excluded(store: MemoryStore) -> None:
    store.insert_belief(_mk("LK", "bananas are always ripe", lock=LOCK_USER))
    assert "LK" not in store.exploration_pool("bananas")


def test_retired_beliefs_are_excluded(store: MemoryStore) -> None:
    ids = _seed(store)
    store.soft_delete_belief(ids[0])
    assert ids[0] not in store.exploration_pool("bananas")


def test_a_belief_with_feedback_is_excluded(store: MemoryStore) -> None:
    """Feedback means it has been shown, so it is not unexplored."""
    ids = _seed(store)
    store.insert_feedback_event(
        ids[0], 1.0, "explicit", "2026-07-31T00:00:00Z"
    )
    pool = store.exploration_pool("bananas")
    assert ids[0] not in pool
    assert ids[1] in pool, "the whole pool collapsed; test would be vacuous"


def test_a_belief_that_was_injected_is_excluded(store: MemoryStore) -> None:
    ids = _seed(store)
    store.record_injection_event(
        session_id="s1",
        turn_id="t1",
        belief_id=ids[0],
        injected_at="2026-07-31T00:00:00Z",
        source="ups",
        active_consumers=[],
    )
    pool = store.exploration_pool("bananas")
    assert ids[0] not in pool
    assert ids[1] in pool, "the whole pool collapsed; test would be vacuous"


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_empty_query_returns_nothing(store: MemoryStore, query: str) -> None:
    """An empty FTS5 MATCH expression raises OperationalError."""
    _seed(store)
    assert store.exploration_pool(query) == []


def test_fts5_special_characters_do_not_raise(store: MemoryStore) -> None:
    """User prompts contain paths, parens and hyphens."""
    _seed(store)
    for q in ["bananas (ripe)", "src/aelfrice/store.py", "a-b OR NEAR", 'q"q']:
        store.exploration_pool(q)


def test_retired_beliefs_are_excluded_even_if_the_fts_row_survives(
    store: MemoryStore,
) -> None:
    """Makes the `valid_to IS NULL` clause load-bearing on its own.

    `soft_delete_belief` prunes the `beliefs_fts` row as well, so the FTS
    join alone already hides a retired belief and the ordinary retire test
    passes with the lifecycle filter removed — verified by mutation. The
    redundancy is deliberate (#980: "index hygiene plus defense-in-depth"),
    and this is the divergence it defends against: the two writes are
    separate statements, so a crash between them leaves a tombstoned belief
    with a live FTS row.
    """
    ids = _seed(store)
    store._conn.execute(
        "UPDATE beliefs SET valid_to = ? WHERE id = ?",
        ("2026-07-31T00:00:00Z", ids[0]),
    )
    store._conn.commit()
    assert store._conn.execute(
        "SELECT COUNT(*) AS n FROM beliefs_fts WHERE id = ?", (ids[0],)
    ).fetchone()["n"] == 1, "fixture did not produce the divergence"
    assert ids[0] not in store.exploration_pool("bananas")


def test_the_pool_is_ordered_by_relevance_not_insertion(
    store: MemoryStore,
) -> None:
    """Pins the `ORDER BY`, which the truncation depends on.

    With `LIMIT` and no `ORDER BY`, which rows survive is the planner's
    choice — in practice insertion order, which would make the pool change
    under a SQLite upgrade with no code change. Verified by mutation that
    dropping the clause fails this and nothing else.

    The strongest match is inserted *last* so relevance order and insertion
    order disagree; a pool that merely echoes insertion order fails here.
    """
    store.insert_belief(_mk("FIRST", "bananas and many other unrelated words"))
    store.insert_belief(_mk("MIDDLE", "bananas with some other words here"))
    store.insert_belief(_mk("BEST", "bananas"))
    pool = store.exploration_pool("bananas")
    assert pool[0] == "BEST", f"got insertion order, not relevance: {pool}"
    assert store.exploration_pool("bananas", limit=1) == ["BEST"]


def test_limit_truncates_deterministically(store: MemoryStore) -> None:
    _seed(store, 8)
    first = store.exploration_pool("bananas", limit=3)
    assert len(first) == 3
    assert first == store.exploration_pool("bananas", limit=3)
    assert store.exploration_pool("bananas", limit=8)[:3] == first


# --- the ledger ------------------------------------------------------------


def test_record_exploration_round_trips(store: MemoryStore) -> None:
    rid = store.record_exploration(
        fire_idx=40,
        seed=0x1234ABCD,
        query="what did we decide about locks",
        candidate_ids=["c1", "c2", "c3"],
        drawn_ids=["c2"],
        displaced_ids=["d9"],
        now="2026-07-31T00:00:00Z",
    )
    row = store._conn.execute(
        "SELECT * FROM exploration_events WHERE id = ?", (rid,)
    ).fetchone()
    assert row["fire_idx"] == 40
    assert json.loads(row["candidate_ids"]) == ["c1", "c2", "c3"]
    assert json.loads(row["drawn_ids"]) == ["c2"]
    assert json.loads(row["displaced_ids"]) == ["d9"]
    assert row["created_at"] == "2026-07-31T00:00:00Z"


def test_a_seed_above_2_63_round_trips(store: MemoryStore) -> None:
    """The reason `seed` is hex TEXT rather than INTEGER.

    SQLite's INTEGER is signed 64-bit. Stored as an integer, a seed at or
    above 2**63 comes back negative, and re-running the draw from the
    recorded row would then silently produce a different result than the
    one the row claims to document.
    """
    seed = (1 << 64) - 1
    rid = store.record_exploration(
        fire_idx=1,
        seed=seed,
        query="what did we decide about locks",
        candidate_ids=[],
        drawn_ids=[],
        displaced_ids=[],
    )
    stored = store._conn.execute(
        "SELECT seed FROM exploration_events WHERE id = ?", (rid,)
    ).fetchone()["seed"]
    assert int(stored, 16) == seed
    assert stored == "ffffffffffffffff"


def test_drawn_id_order_is_preserved_not_sorted(store: MemoryStore) -> None:
    """Draw order records which slot each belief filled."""
    rid = store.record_exploration(
        fire_idx=1,
        seed=1,
        query="what did we decide about locks",
        candidate_ids=["a", "b", "c"],
        drawn_ids=["c", "a"],
        displaced_ids=[],
    )
    row = store._conn.execute(
        "SELECT drawn_ids FROM exploration_events WHERE id = ?", (rid,)
    ).fetchone()
    assert json.loads(row["drawn_ids"]) == ["c", "a"]


def test_the_ledger_survives_the_belief_it_names(store: MemoryStore) -> None:
    """No foreign key, on purpose.

    The row documents a decision taken while the belief was live. A later
    retire or hard delete must not cascade the record away, or the audit
    trail thins out exactly where someone is asking what happened.
    """
    ids = _seed(store)
    store.record_exploration(
        fire_idx=20,
        seed=7,
        query="what did we decide about locks",
        candidate_ids=ids,
        drawn_ids=[ids[0]],
        displaced_ids=[],
    )
    store.delete_belief(ids[0])
    rows = store._conn.execute("SELECT * FROM exploration_events").fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0]["drawn_ids"]) == [ids[0]]


def test_rows_accumulate_rather_than_replace(store: MemoryStore) -> None:
    for i in range(3):
        store.record_exploration(
            fire_idx=20 * i,
            seed=i,
            query="what did we decide about locks",
            candidate_ids=[],
            drawn_ids=[],
            displaced_ids=[],
        )
    n = store._conn.execute(
        "SELECT COUNT(*) AS n FROM exploration_events"
    ).fetchone()["n"]
    assert n == 3


def test_the_raw_query_never_reaches_the_table(tmp_path: Path) -> None:
    """The column's reason is privacy, so pin it as behaviour.

    `record_exploration` takes the query and hashes it internally, so
    there is no parameter a caller can hand the raw prompt to. This
    asserts the prompt is absent from the stored row *and* that the
    stored value is the digest — the first alone would pass against a
    column that stored nothing useful.
    """
    prompt = "the user's private prompt about acme corp revenue"
    store = MemoryStore(str(tmp_path / "m.db"))
    try:
        store.record_exploration(
            fire_idx=1,
            seed=7,
            query=prompt,
            candidate_ids=["a"],
            drawn_ids=["a"],
            displaced_ids=[],
        )
        row = store._conn.execute(
            "SELECT query_hash FROM exploration_events"
        ).fetchone()
        assert prompt not in row["query_hash"]
        assert "acme" not in row["query_hash"]
        assert row["query_hash"] == hash_query(prompt)
        assert len(row["query_hash"]) == 16
    finally:
        store.close()


def test_hash_query_is_stable_and_distinguishing(tmp_path: Path) -> None:
    """A constant would satisfy the test above; this rules it out."""
    assert hash_query("a") == hash_query("a")
    assert hash_query("a") != hash_query("b")
