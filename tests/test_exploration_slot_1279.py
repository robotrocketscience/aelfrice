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

from aelfrice.exploration import DEFAULT_EXPLORATION_CADENCE
from aelfrice.hook import _substitute_exploration_slots
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, LOCK_USER, Belief
from aelfrice.store import SCHEMA_META_EXPLORATION_FIRE_IDX, MemoryStore

_SESSION = "sess-1279"

# Fire indices are derived from the shipped cadence rather than written as
# literals: these were `20`/`21`/`40` and every one of them silently stopped
# exercising its branch when the default moved (#1279 review). Multiples so
# the turn fires; `+ 1` so it does not.
_FIRING = DEFAULT_EXPLORATION_CADENCE * 4
_FIRING_LATER = DEFAULT_EXPLORATION_CADENCE * 8
_NOT_FIRING = _FIRING + 1


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


def _fire(store: MemoryStore, idx: int) -> None:
    """Arm the *store* counter so the next claim lands on `idx` (#1294).

    Was a `monkeypatch` of `session_ring.read_ring_state`. The fire index
    is now store-level rather than per-session, so pinning the ring would
    pin nothing — every one of these tests would silently stop exercising
    its branch. Seeds `idx - 1`, because `next_exploration_fire_idx` is
    1-based and returns the post-increment value.
    """
    store._conn.execute(
        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
        (SCHEMA_META_EXPLORATION_FIRE_IDX, str(idx - 1)),
    )
    store._conn.commit()


def _run(store, hits, capsys, *, query="exploration probe widget", cwd=None):
    import sys

    out = _substitute_exploration_slots(
        hits, session_id=_SESSION, query=query, store=store, serr=sys.stderr,
        cwd=cwd,
    )
    capsys.readouterr()
    return out


# --- the flag ------------------------------------------------------------


def test_default_off_is_a_no_op(store, monkeypatch, capsys) -> None:
    """Nothing changes until the operator opts in."""
    _seed_pool(store)
    _fire(store, _FIRING)
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
    _fire(store, _FIRING)
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
    """Cadence has to actually gate: one index off a multiple."""
    _seed_pool(store)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    hits = [_mk(f"h{i}", f"ranked hit {i} " + "filler " * 8) for i in range(4)]

    _fire(store, _NOT_FIRING)
    assert _run(store, hits, capsys) == hits
    _fire(store, _FIRING)
    assert _run(store, hits, capsys) != hits


# --- the invariants ------------------------------------------------------


def test_a_user_lock_is_never_displaced(store, monkeypatch, capsys) -> None:
    """An all-locked pack is a no-op, not an eviction.

    L0 is injected unconditionally. If exploration could take a lock's slot
    it would be silently overriding the one tier the user set by hand.
    """
    _seed_pool(store)
    _fire(store, _FIRING)
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
    _fire(store, _FIRING)
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
    _fire(store, _FIRING)
    hits = [_mk(f"h{i}", f"ranked hit {i} " + "filler " * 8) for i in range(6)]

    before = sum(_belief_tokens(b) for b in _run(store, hits, capsys))
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    after = sum(_belief_tokens(b) for b in _run(store, hits, capsys))

    assert after <= before


def test_the_draw_is_deterministic(store, monkeypatch, capsys) -> None:
    """Same (session, fire_idx, query, pool) -> same belief. Replay needs it.

    The counter is re-armed between the two arms because it now advances
    on every consultation (#1294) — previously the ring was patched to a
    constant, so repeated calls saw the same index for free. Re-arming is
    what makes this a test of the *seed* rather than of the counter.
    """
    _seed_pool(store)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    hits = [_mk(f"h{i}", f"ranked hit {i} " + "filler " * 8) for i in range(6)]

    _fire(store, _FIRING)
    first = [b.id for b in _run(store, hits, capsys)]
    _fire(store, _FIRING)
    second = [b.id for b in _run(store, hits, capsys)]
    assert first == second

    # ...and a different turn draws a different belief, or "deterministic"
    # would be satisfied by always drawing the same one.
    _fire(store, _FIRING_LATER)
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
    _fire(store, _FIRING)
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
    _fire(store, _FIRING)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")

    def _boom(*a, **k):
        raise RuntimeError("pool query exploded")

    monkeypatch.setattr(MemoryStore, "exploration_pool", _boom)
    hits = [_mk("h1", "ranked hit one"), _mk("h2", "ranked hit two")]
    assert _run(store, hits, capsys) == hits


def test_an_empty_pool_is_a_no_op(store, monkeypatch, capsys) -> None:
    """Nothing to explore is not an error, and must not empty the pack."""
    _fire(store, _FIRING)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")
    hits = [_mk("h1", "ranked hit one"), _mk("h2", "ranked hit two")]
    assert _run(store, hits, capsys) == hits


def test_a_belief_already_in_the_pack_is_not_drawn_again(
    store, monkeypatch, capsys,
) -> None:
    """The pool is filtered against the pack, so a slot cannot duplicate."""
    pool_ids = _seed_pool(store)
    _fire(store, _FIRING)
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
    _fire(store, _FIRING)
    monkeypatch.setenv("AELFRICE_EXPLORATION", "1")

    hits = [_mk("h1", "tiny"), _mk("h2", "also tiny")]
    assert sum(_belief_tokens(b) for b in hits) < _belief_tokens(
        store.get_belief("pool00")
    )
    assert _run(store, hits, capsys) == hits


# --- the resolvers -------------------------------------------------------
#
# The suite above reaches these only through `AELFRICE_EXPLORATION`, so the
# precedence chain and the disable contract were asserted by the docs and by
# nothing else. Each test below pins one tier against the tier beneath it, so
# a collapsed precedence fails rather than merely changing which default wins.


def _toml(tmp_path: Path, body: str) -> Path:
    (tmp_path / ".aelfrice.toml").write_text(f"[retrieval]\n{body}\n")
    return tmp_path


def test_exploration_flag_precedence(monkeypatch, tmp_path) -> None:
    from aelfrice.retrieval import is_exploration_enabled

    assert is_exploration_enabled(start=tmp_path) is False
    _toml(tmp_path, "exploration_enabled = true")
    assert is_exploration_enabled(start=tmp_path) is True
    # kwarg loses to TOML? No — kwarg outranks it, and False must be
    # honoured rather than treated as "unset", which is the classic
    # tri-state bug.
    assert is_exploration_enabled(False, start=tmp_path) is False
    monkeypatch.setenv("AELFRICE_EXPLORATION", "0")
    assert is_exploration_enabled(True, start=tmp_path) is False


def test_cadence_precedence(monkeypatch, tmp_path) -> None:
    from aelfrice.retrieval import (
        DEFAULT_EXPLORATION_CADENCE,
        resolve_exploration_cadence,
    )

    assert resolve_exploration_cadence(start=tmp_path) == (
        DEFAULT_EXPLORATION_CADENCE
    )
    _toml(tmp_path, "exploration_cadence = 7")
    assert resolve_exploration_cadence(start=tmp_path) == 7
    assert resolve_exploration_cadence(3, start=tmp_path) == 3
    monkeypatch.setenv("AELFRICE_EXPLORATION_CADENCE", "11")
    assert resolve_exploration_cadence(3, start=tmp_path) == 11


def test_env_cadence_zero_disables_rather_than_falling_through(
    monkeypatch, tmp_path
) -> None:
    """The documented contract is "`0` or less disables". On the env tier it
    did the opposite: `_env_positive_int` discarded the value and the caller
    got `DEFAULT_EXPLORATION_CADENCE`, so an operator asking for "off" got
    exploration every 20th turn. Distinguishing against the default is the
    whole point — asserting `<= 0` alone would pass against the bug.
    """
    from aelfrice.exploration import should_explore
    from aelfrice.retrieval import (
        DEFAULT_EXPLORATION_CADENCE,
        resolve_exploration_cadence,
    )

    for raw in ("0", "-1"):
        monkeypatch.setenv("AELFRICE_EXPLORATION_CADENCE", raw)
        got = resolve_exploration_cadence(start=tmp_path)
        assert got == int(raw)
        assert got != DEFAULT_EXPLORATION_CADENCE
        # And it actually disables, at the fire index that would otherwise
        # be the firing one.
        assert should_explore(DEFAULT_EXPLORATION_CADENCE, cadence=got) is False


def test_a_non_numeric_cadence_falls_through_to_the_default(
    monkeypatch, tmp_path, capsys
) -> None:
    from aelfrice.retrieval import (
        DEFAULT_EXPLORATION_CADENCE,
        resolve_exploration_cadence,
    )

    monkeypatch.setenv("AELFRICE_EXPLORATION_CADENCE", "sometimes")
    assert resolve_exploration_cadence(start=tmp_path) == (
        DEFAULT_EXPLORATION_CADENCE
    )
    assert "expected int" in capsys.readouterr().err


def test_slots_precedence_and_that_zero_is_still_rejected(
    monkeypatch, tmp_path
) -> None:
    """Slots keeps `_env_positive_int`: zero slots is not a disable knob,
    it is a meaningless pack request, and `exploration_enabled` is the
    documented off switch. Pinned so the cadence fix is not copied here by
    symmetry.
    """
    from aelfrice.retrieval import (
        DEFAULT_EXPLORATION_SLOTS,
        resolve_exploration_slots,
    )

    assert resolve_exploration_slots(start=tmp_path) == DEFAULT_EXPLORATION_SLOTS
    _toml(tmp_path, "exploration_slots = 4")
    assert resolve_exploration_slots(start=tmp_path) == 4
    assert resolve_exploration_slots(2, start=tmp_path) == 2
    monkeypatch.setenv("AELFRICE_EXPLORATION_SLOTS", "0")
    assert resolve_exploration_slots(start=tmp_path) == 4


def test_the_cadence_is_reachable_and_means_one_turn_in_n() -> None:
    """Supersedes the per-session reachability guard (#1279 -> #1294).

    The old version asserted that a session of typical length reaches a
    firing turn, because `fire_idx` came from the session ring and the
    knob really did mean "one turn in n *of a session*". That property is
    the wrong one now: the counter is store-level, so what has to hold is
    the knob's stated meaning — one fire per `cadence` turns, however the
    turns are distributed across sessions.

    Deliberately not `assert DEFAULT_EXPLORATION_CADENCE == 20`, which
    would be a tautology. Dropping the modulus guard, or reverting to a
    counter that restarts, breaks the count here.
    """
    from aelfrice.exploration import DEFAULT_EXPLORATION_CADENCE, should_explore

    cadence = DEFAULT_EXPLORATION_CADENCE
    turns = cadence * 5
    fires = [i for i in range(1, turns + 1) if should_explore(i, cadence=cadence)]
    assert len(fires) == 5, f"{len(fires)} fires over {turns} turns at cadence {cadence}"
    assert fires == [cadence * k for k in range(1, 6)]


def test_the_fire_index_accumulates_across_sessions(tmp_path) -> None:
    """AC: a second session sees the first session's count (#1294).

    This is the whole point of the issue. Under the session ring the
    second store instance restarted at the beginning, so a cadence above
    the typical session length was unreachable forever. Two *store
    instances* over one file stand in for two sessions, which is exactly
    what the ring could not do — it keyed on session id and returned
    `{}` on a mismatch.
    """
    db = str(tmp_path / "memory.db")

    first = MemoryStore(db)
    a = [first.next_exploration_fire_idx() for _ in range(3)]
    first.close()

    second = MemoryStore(db)
    b = [second.next_exploration_fire_idx() for _ in range(3)]
    second.close()

    assert a == [1, 2, 3]
    assert b == [4, 5, 6], "the counter restarted — this is the #1294 defect"
    assert min(b) > max(a)


def test_sequential_claims_are_unique_monotonic_and_gapless(tmp_path) -> None:
    """Two store handles over one file never repeat or skip an index.

    This pins the increment arithmetic. It does **not** pin atomicity —
    each claim here completes before the next begins, so it stays green
    with the transaction downgraded to deferred. That gap is covered by
    the test below; keeping the two separate so neither is mistaken for
    the other.
    """
    db = str(tmp_path / "memory.db")
    one, two = MemoryStore(db), MemoryStore(db)
    try:
        claimed = []
        for _ in range(5):
            claimed.append(one.next_exploration_fire_idx())
            claimed.append(two.next_exploration_fire_idx())
        assert claimed == list(range(1, 11))
    finally:
        one.close()
        two.close()


def test_the_claim_takes_the_write_lock_before_reading(tmp_path) -> None:
    """AC: concurrent advancement is atomic (#1294).

    The read-then-write has to run under `BEGIN IMMEDIATE`. Deferred, two
    sister sessions sharing the store both pass the SELECT before either
    UPDATEs, both compute the same successor, and `exploration_events`
    gains two rows claiming to be the same draw — with the same seed, so
    replay cannot tell them apart.

    Asserts the lock discipline directly rather than trying to stage a
    race: a sequential test cannot observe the interleaving, and a
    threaded one would trade a real assertion for a flaky one. Dropping
    `immediate=True` turns this red, which a behavioural test at this
    level could not manage.
    """
    store = MemoryStore(str(tmp_path / "memory.db"))
    seen: list[bool] = []
    real = store.transaction

    def spy(*args, **kwargs):
        seen.append(bool(kwargs.get("immediate", False)))
        return real(*args, **kwargs)

    try:
        store.transaction = spy  # type: ignore[method-assign]
        store.next_exploration_fire_idx()
    finally:
        store.transaction = real  # type: ignore[method-assign]
        store.close()

    assert seen == [True], (
        "the fire-index claim did not open an immediate transaction, so two "
        "sessions can be handed the same index"
    )


def test_the_project_toml_is_read_from_the_payload_cwd(
    store, monkeypatch, capsys, tmp_path,
) -> None:
    """`[retrieval] exploration_enabled` has to resolve against the payload's
    project directory, not the hook process's working directory.

    Every other config-dependent step in `user_prompt_submit` threads
    `payload_cwd` — `load_user_prompt_submit_config(start=payload_cwd)` and
    both phantom configs. This one did not, so a project that opted in via
    its own `.aelfrice.toml` got the slot only when the hook happened to be
    running from that directory (#1285 review).

    Distinguishing by construction: the same store, query and ring counter
    are run twice and differ *only* in `cwd`, so dropping `start=cwd` from
    the resolvers collapses the two arms and this goes red.
    """
    project = tmp_path / "project"
    project.mkdir()
    (project / ".aelfrice.toml").write_text(
        "[retrieval]\nexploration_enabled = true\n", encoding="utf-8"
    )

    _seed_pool(store)
    _fire(store, _FIRING)
    hits = [_mk(f"h{i}", f"ranked hit {i} " + "filler " * 8) for i in range(4)]

    # The autouse fixture has chdir'd to `tmp_path`, which does not contain
    # the TOML, so the no-cwd arm cannot see the opt-in.
    without_cwd = _run(store, hits, capsys)
    with_cwd = _run(store, hits, capsys, cwd=project)

    assert without_cwd == hits, "the opt-in leaked in without a cwd"
    assert [b.id for b in with_cwd] != [b.id for b in hits]
    assert any(b.id.startswith("pool") for b in with_cwd)
