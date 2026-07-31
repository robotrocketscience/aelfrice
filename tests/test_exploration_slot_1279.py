"""The exploration slot's consumer (#1279, #1176 proposal 5).

Every test here asserts an outcome that **differs from the exploration-off
arm**. That is deliberate rather than stylistic: the pool query, the ledger
and the seeded draw were all merged before any of them had a caller, and a
test that merely counted hits would have passed against that no-op exactly as
happily as against a working slot. Three separate mechanisms on #1176 shipped
inert; a counting test is how that goes unnoticed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aelfrice.hook import _substitute_exploration_slots
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import MemoryStore

_SESSION = "sess-1279"


def _mk(bid: str, content: str, lock_level: str = LOCK_NONE) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        created_at="2026-07-31T00:00:00Z",
        last_retrieved_at=None,
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Pin every ambient tier so the resolvers cannot read the real config.

    `AELFRICE_EXPLORATION` unset plus a chdir into `tmp_path` means the TOML
    walk cannot reach the repo's own `.aelfrice.toml`; without this the suite
    would be green or red depending on the developer's environment.
    """
    for var in (
        "AELFRICE_EXPLORATION",
        "AELFRICE_EXPLORATION_CADENCE",
        "AELFRICE_EXPLORATION_SLOTS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    yield


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(str(tmp_path / "memory.db"))
    yield s
    s.close()


def _seed_pool(store: MemoryStore, n: int = 12) -> list[str]:
    """Beliefs that match the query and have never been injected."""
    ids = []
    for i in range(n):
        b = _mk(f"pool{i:02d}", f"exploration probe belief {i} widget alpha")
        store.insert_belief(b)
        ids.append(b.id)
    return ids


def _fire(monkeypatch: pytest.MonkeyPatch, idx: int) -> None:
    """Pin the ring counter so the cadence decision is deterministic."""
    monkeypatch.setattr(
        "aelfrice.session_ring.read_ring_state",
        lambda sid: {"next_fire_idx": idx},
    )


def _run(store, hits, capsys, *, query="exploration probe widget"):
    import sys

    out = _substitute_exploration_slots(
        hits, session_id=_SESSION, query=query, store=store, serr=sys.stderr,
    )
    capsys.readouterr()
    return out


# --- the flag ------------------------------------------------------------


def test_default_off_is_a_no_op(store, monkeypatch, capsys) -> None:
    """Nothing changes until the operator opts in."""
    _seed_pool(store)
    _fire(monkeypatch, 20)
    hits = [_mk("h1", "ranked hit one"), _mk("h2", "ranked hit two")]
    assert _run(store, hits, capsys) == hits


def test_enabled_on_a_firing_turn_changes_the_pack(
    store, monkeypatch, capsys,
) -> None:
    """The distinguishing case: same inputs, on vs off, different pack.

    Without this the whole suite could pass against a consumer that returns
    its input.
    """
    _seed_pool(store)
    _fire(monkeypatch, 20)
    hits = [_mk(f"h{i}", f"ranked hit {i} " + "filler " * 8) for i in range(4)]

    off = _run(store, hits, capsys)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    on = _run(store, hits, capsys)

    assert off == hits
    assert [b.id for b in on] != [b.id for b in off]
    assert any(b.id.startswith("pool") for b in on)


def test_non_firing_turn_is_a_no_op_even_when_enabled(
    store, monkeypatch, capsys,
) -> None:
    """Cadence has to actually gate. 21 % 20 != 0."""
    _seed_pool(store)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    hits = [_mk(f"h{i}", f"ranked hit {i} " + "filler " * 8) for i in range(4)]

    _fire(monkeypatch, 21)
    assert _run(store, hits, capsys) == hits
    _fire(monkeypatch, 20)
    assert _run(store, hits, capsys) != hits


# --- the invariants ------------------------------------------------------


def test_a_user_lock_is_never_displaced(store, monkeypatch, capsys) -> None:
    """An all-locked pack is a no-op, not an eviction.

    L0 is injected unconditionally. If exploration could take a lock's slot
    it would be silently overriding the one tier the user set by hand.
    """
    _seed_pool(store)
    _fire(monkeypatch, 20)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    hits = [
        _mk("L1", "a locked belief", LOCK_USER),
        _mk("L2", "another locked belief", LOCK_USER),
    ]
    assert _run(store, hits, capsys) == hits


def test_locks_survive_while_the_non_locked_tail_is_displaced(
    store, monkeypatch, capsys,
) -> None:
    """Mixed pack: the lock stays, the non-locked tail pays.

    Paired with the all-locked test above so "locks survive" cannot pass by
    the slot never firing at all.
    """
    _seed_pool(store)
    _fire(monkeypatch, 20)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    hits = [
        _mk("L1", "a locked belief", LOCK_USER),
        _mk("n1", "a non locked hit with a reasonable amount of text here"),
    ]
    out = _run(store, hits, capsys)

    assert "L1" in [b.id for b in out]
    assert any(b.id.startswith("pool") for b in out)
    assert "n1" not in [b.id for b in out]


def test_the_block_does_not_grow_in_tokens(store, monkeypatch, capsys) -> None:
    """Substitution, not append — measured on tokens, not on cardinality.

    A drawn belief can be longer than the hit it replaces, so a 1-for-1 swap
    would not be enough. The slot must free at least what it spends, or it is
    a budget increase in disguise and its own coverage measurement is
    confounded.
    """
    from aelfrice.retrieval import _belief_tokens

    _seed_pool(store)
    _fire(monkeypatch, 20)
    hits = [_mk(f"h{i}", f"ranked hit {i} " + "filler " * 8) for i in range(6)]

    before = sum(_belief_tokens(b) for b in _run(store, hits, capsys))
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    after = sum(_belief_tokens(b) for b in _run(store, hits, capsys))

    assert after <= before


def test_the_draw_is_deterministic(store, monkeypatch, capsys) -> None:
    """Same (session, fire_idx, query, pool) -> same belief. Replay needs it."""
    _seed_pool(store)
    _fire(monkeypatch, 20)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    hits = [_mk(f"h{i}", f"ranked hit {i} " + "filler " * 8) for i in range(6)]

    first = [b.id for b in _run(store, hits, capsys)]
    second = [b.id for b in _run(store, hits, capsys)]
    assert first == second

    # ...and a different turn draws a different belief, or "deterministic"
    # would be satisfied by always drawing the same one.
    _fire(monkeypatch, 40)
    third = [b.id for b in _run(store, hits, capsys)]
    assert {i for i in third if i.startswith("pool")} != {
        i for i in first if i.startswith("pool")
    }


def test_a_firing_turn_writes_one_ledger_row_naming_what_it_evicted(
    store, monkeypatch, capsys,
) -> None:
    """The ledger is what makes the slot auditable, so it records the eviction.

    Recording only the drawn id would make it impossible to reconstruct what
    the pack would have been, which is exactly the counterfactual any coverage
    analysis needs.
    """
    _seed_pool(store)
    _fire(monkeypatch, 20)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    hits = [_mk(f"h{i}", f"ranked hit {i} " + "filler " * 8) for i in range(6)]

    out = _run(store, hits, capsys)
    rows = store._conn.execute(
        "SELECT drawn_ids, displaced_ids FROM exploration_events"
    ).fetchall()

    assert len(rows) == 1
    drawn_ids, displaced_ids = rows[0][0], rows[0][1]
    assert any(b.id in drawn_ids for b in out if b.id.startswith("pool"))
    # Something was evicted, and it is named.
    evicted = {b.id for b in hits} - {b.id for b in out}
    assert evicted
    for eid in evicted:
        assert eid in displaced_ids


# --- fail-soft -----------------------------------------------------------


def test_a_raising_pool_query_leaves_the_pack_untouched(
    store, monkeypatch, capsys,
) -> None:
    """A research lane must never be why a hook fails."""
    _fire(monkeypatch, 20)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")

    def _boom(*a, **k):
        raise RuntimeError("pool query exploded")

    monkeypatch.setattr(MemoryStore, "exploration_pool", _boom)
    hits = [_mk("h1", "ranked hit one"), _mk("h2", "ranked hit two")]
    assert _run(store, hits, capsys) == hits


def test_an_empty_pool_is_a_no_op(store, monkeypatch, capsys) -> None:
    """Nothing to explore is not an error, and must not empty the pack."""
    _fire(monkeypatch, 20)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    hits = [_mk("h1", "ranked hit one"), _mk("h2", "ranked hit two")]
    assert _run(store, hits, capsys) == hits


def test_a_belief_already_in_the_pack_is_not_drawn_again(
    store, monkeypatch, capsys,
) -> None:
    """The pool is filtered against the pack, so a slot cannot duplicate."""
    pool_ids = _seed_pool(store)
    _fire(monkeypatch, 20)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")

    hits = [store.get_belief(i) for i in pool_ids]
    out = _run(store, hits, capsys)
    ids = [b.id for b in out]
    assert len(ids) == len(set(ids))


def test_a_pack_too_cheap_to_pay_for_the_draw_is_left_alone(
    store, monkeypatch, capsys,
) -> None:
    """When the non-locked tail cannot fund the draw, skip rather than grow.

    This is the token guard's other half and it is easy to lose: the obvious
    implementation swaps one hit for one belief, which grows the block
    whenever the drawn belief is longer. Here the whole pack is worth fewer
    tokens than the belief being drawn, so the only options are "grow" or
    "do nothing", and it must do nothing.
    """
    from aelfrice.retrieval import _belief_tokens

    _seed_pool(store)
    _fire(monkeypatch, 20)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")

    hits = [_mk("h1", "tiny"), _mk("h2", "also tiny")]
    assert sum(_belief_tokens(b) for b in hits) < _belief_tokens(
        store.get_belief("pool00")
    )
    assert _run(store, hits, capsys) == hits
