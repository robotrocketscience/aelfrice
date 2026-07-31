"""#1176 proposal 3 — ACT-R fan effect on the L2.5 entity lane.

The shipped lane ranks by `COUNT(DISTINCT entity_lower)`, which prices
every matched entity the same. It is not: on the live store `tmp` sits in
1,480 beliefs and 86% of entities sit in exactly one, so a match on a
corpus-ubiquitous token buys the same rank as a match on a unique symbol
— on a lane that holds unconditional budget precedence.

The fixtures below are built so the *count* lane and the *fan* lane
disagree, and so that the disagreement can only come from fan: the
rare-matching belief is given the alphabetically **later** id, so under
the shipped overlap/id ordering it always loses. If the fan weighting
were dropped, every ordering assertion here reverts to id order.

Default-off is byte-identical, and the resolver is pinned against the
#1107 production gap — `retrieve()`, not `retrieve_v2`, is what the
hooks call, and a lane wired only into the latter is inert on the path
the A/B has to measure.
"""
from __future__ import annotations

import math

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_USER_TRANSCRIPT,
    ORIGIN_USER_VALIDATED,
    Belief,
)
from aelfrice.retrieval import ENV_FAN_EFFECT, is_fan_effect_enabled, retrieve
from aelfrice.store import MemoryStore

#: The live high-fan entities are bare tokens (`tmp`, `and`, `pr`). A
#: path is used here because `test_env_var_reaches_the_production_retrieve_path`
#: drives the real query-side extractor, which only keys on shapes it
#: recognises — bare `tmp` extracts to nothing and the end-to-end arm
#: would silently measure a one-entity query. The lane arithmetic is
#: identical either way; only the query-side extraction differs.
UBIQUITOUS = "src/common.py"
RARE = "src/widget.py"
#: Enough filler beliefs carrying UBIQUITOUS that its fan dominates.
N_FILLER = 12


def _mk(bid: str, content: str, origin: str = ORIGIN_USER_TRANSCRIPT) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-07-30T00:00:00Z",
        last_retrieved_at=None,
        origin=origin,
    )


def _entities(store: MemoryStore, bid: str, *lowers: str) -> None:
    """Replace a belief's auto-extracted entities with an exact set."""
    store._conn.execute(
        "DELETE FROM belief_entities WHERE belief_id=?", (bid,)
    )
    for lower in lowers:
        store._conn.execute(
            "INSERT INTO belief_entities(belief_id, entity_lower, "
            "entity_raw, kind, span_start, span_end) VALUES (?,?,?,?,0,0)",
            (bid, lower, lower, "identifier"),
        )
    store._conn.commit()


def _seed(store: MemoryStore) -> None:
    """Two candidates at overlap 1 apiece, plus fan for the common term.

    `aaa_common` wins on id, `zzz_rare` wins on fan. Nothing else
    separates them, so any ordering difference is the fan weighting.
    """
    store.insert_belief(_mk("aaa_common", "one"))
    _entities(store, "aaa_common", UBIQUITOUS)
    store.insert_belief(_mk("zzz_rare", "two"))
    _entities(store, "zzz_rare", RARE)
    for i in range(N_FILLER):
        bid = f"filler_{i:02d}"
        store.insert_belief(_mk(bid, f"filler {i}"))
        _entities(store, bid, UBIQUITOUS)


@pytest.fixture()
def store():
    s = MemoryStore(":memory:")
    _seed(s)
    yield s
    s.close()


def _order(store: MemoryStore, *, fan: bool, limit: int = 10,
           origin_tiebreak: bool = False, keys=None) -> list[str]:
    return [
        bid
        for bid, _ in store.lookup_entities(
            keys if keys is not None else [UBIQUITOUS, RARE],
            limit=limit, fan_effect=fan, origin_tiebreak=origin_tiebreak,
        )
    ]


# --- the reorder ---------------------------------------------------------


def test_fan_lifts_the_rare_match_over_the_ubiquitous_one(store) -> None:
    """The whole proposal in one assertion.

    Both candidates match exactly one query entity, so the count lane
    cannot separate them and falls through to id. Fan can: `tmp` is
    carried by 13 beliefs here and `src/widget.py` by one.
    """
    ranked = _order(store, fan=True)
    assert ranked.index("zzz_rare") < ranked.index("aaa_common")


def test_the_two_lanes_actually_disagree_on_this_fixture(store) -> None:
    """Control. Without this, every byte-identical assertion below could
    pass against a fixture the reorder never touches."""
    assert _order(store, fan=False)[0] == "aaa_common"
    assert _order(store, fan=True)[0] == "zzz_rare"


def test_off_is_byte_identical(store) -> None:
    """Default, explicit-off and the pre-#1176 ordering are one list."""
    default = _order(store, fan=False)
    assert default == [
        bid for bid, _ in store.lookup_entities(
            [UBIQUITOUS, RARE], limit=10,
        )
    ]
    assert default[0] == "aaa_common"
    assert default == sorted(default)  # pure id order at overlap 1


def test_limit_is_applied_after_the_reorder(store) -> None:
    """A limit pushed down to SQL would truncate to the *count* lane's
    top row and the reorder would be invisible at limit=1 — which is
    where a lane with a tight sub-budget actually operates."""
    assert _order(store, fan=True, limit=1) == ["zzz_rare"]


def test_overlap_count_is_unchanged_by_the_reorder(store) -> None:
    """The second tuple element stays the overlap count, not the
    activation: this changes the lane's order, not its interface."""
    got = dict(store.lookup_entities([UBIQUITOUS, RARE], limit=10,
                                     fan_effect=True))
    assert got["zzz_rare"] == 1
    assert got["aaa_common"] == 1


def test_more_matches_never_lower_activation(store) -> None:
    """Every `S_ji` term is non-negative (`fan_j <= N + 1`), so carrying
    an extra query entity cannot demote a belief. A sign slip in the
    log would invert exactly this."""
    store.insert_belief(_mk("mmm_both", "three"))
    _entities(store, "mmm_both", UBIQUITOUS, RARE)
    ranked = _order(store, fan=True)
    assert ranked[0] == "mmm_both"


def test_equal_fan_degenerates_to_the_count_lane() -> None:
    """With all fans equal the activation is a constant multiple of the
    overlap, so the fan lane must reproduce the count lane exactly. This
    is what distinguishes 'weighted by fan' from 'reordered somehow'."""
    s = MemoryStore(":memory:")
    try:
        s.insert_belief(_mk("aaa", "one"))
        _entities(s, "aaa", "alpha")
        s.insert_belief(_mk("bbb", "two"))
        _entities(s, "bbb", "beta")
        s.insert_belief(_mk("ccc", "three"))
        _entities(s, "ccc", "alpha", "beta")
        keys = ["alpha", "beta"]
        assert _order(s, fan=True, keys=keys) == _order(s, fan=False, keys=keys)
        assert _order(s, fan=True, keys=keys)[0] == "ccc"  # overlap 2 leads
    finally:
        s.close()


def test_retired_beliefs_do_not_inflate_fan(store) -> None:
    """Fan counts *active* beliefs, matching the lane's own `valid_to`
    filter. Retire the filler and `tmp` is no longer common, so the
    reorder must relax back to id order — a fan count that ignored
    `valid_to` would keep damping a term nothing carries any more."""
    for i in range(N_FILLER):
        store.soft_delete_belief(f"filler_{i:02d}")
    assert _order(store, fan=True)[0] == "aaa_common"


def test_ties_are_broken_by_id_not_by_row_order(store) -> None:
    """Two beliefs carrying the same entity set land on bit-identical
    activations, so the id tie-break is what makes the order total."""
    store.insert_belief(_mk("bbb_twin", "twin"))
    _entities(store, "bbb_twin", RARE)
    ranked = _order(store, fan=True)
    assert ranked[:2] == ["bbb_twin", "zzz_rare"]


def test_origin_tiebreak_composes_with_fan(store) -> None:
    """An activation tie still routes through the #1089 origin priority
    before id, so the two lanes stack rather than one shadowing the
    other."""
    store.insert_belief(_mk("bbb_twin", "twin", ORIGIN_USER_TRANSCRIPT))
    _entities(store, "bbb_twin", RARE)
    store.insert_belief(_mk("ccc_curated", "curated", ORIGIN_USER_VALIDATED))
    _entities(store, "ccc_curated", RARE)
    ranked = _order(store, fan=True, origin_tiebreak=True)
    assert ranked[0] == "ccc_curated"
    # …and without it, the same three fall back to plain id order.
    assert _order(store, fan=True)[:3] == [
        "bbb_twin", "ccc_curated", "zzz_rare",
    ]


def test_ordering_is_deterministic_across_calls(store) -> None:
    assert _order(store, fan=True) == _order(store, fan=True)


def test_activation_matches_the_closed_form(store) -> None:
    """Pin the formula itself, so a refactor that keeps the *ordering*
    while changing the scale is still visible. With N active beliefs,
    a belief matching one entity of fan f scores ln((N+1)/(f+1)); the
    ranked gap between the two candidates is the difference."""
    n_active = store.count_active_beliefs()
    rare_fan, common_fan = 1, N_FILLER + 1
    rare = math.log(n_active + 1.0) - math.log(rare_fan + 1.0)
    common = math.log(n_active + 1.0) - math.log(common_fan + 1.0)
    assert rare > common >= 0.0


def test_empty_and_unknown_keys_are_inert(store) -> None:
    assert store.lookup_entities([], limit=10, fan_effect=True) == []
    assert store.lookup_entities(["nothing"], limit=10, fan_effect=True) == []
    assert store.lookup_entities([RARE], limit=0, fan_effect=True) == []


# --- resolver + production reachability ----------------------------------


def test_resolver_defaults_off_and_env_beats_kwarg(monkeypatch) -> None:
    monkeypatch.delenv(ENV_FAN_EFFECT, raising=False)
    assert is_fan_effect_enabled() is False
    assert is_fan_effect_enabled(True) is True
    monkeypatch.setenv(ENV_FAN_EFFECT, "1")
    assert is_fan_effect_enabled(False) is True
    monkeypatch.setenv(ENV_FAN_EFFECT, "0")
    assert is_fan_effect_enabled(True) is False
    monkeypatch.setenv(ENV_FAN_EFFECT, "not-a-bool")
    assert is_fan_effect_enabled() is False


def test_env_var_reaches_the_production_retrieve_path(store, monkeypatch) -> None:
    """#1107 guard: drive `retrieve()`, which is what the hooks call.

    A lane threaded into `retrieve_v2` but not through
    `retrieve_with_tiers` -> `_l25_hits` -> `lookup_entities` is inert on
    the path the A/B measures, and the A/B would then report no
    difference because no treatment ever ran. Dropping the wiring at any
    of those three joints turns this red.

    What it deliberately does **not** claim: that `retrieve()` must spell
    the kwarg `None` rather than `False`. The resolver is env-first, so
    `AELFRICE_FAN_EFFECT=1` overrides either spelling — hard-coding
    `False` at that call site leaves this whole file green. `None` is
    still the right spelling (a `.aelfrice.toml` tier would need it), but
    it is a convention here, not a guard, and asserting otherwise would
    be pinning a claim the code does not make.
    """
    query = f"{UBIQUITOUS} {RARE}"
    monkeypatch.delenv(ENV_FAN_EFFECT, raising=False)
    off = [b.id for b in retrieve(store, query, token_budget=4000)]
    monkeypatch.setenv(ENV_FAN_EFFECT, "1")
    on = [b.id for b in retrieve(store, query, token_budget=4000)]
    assert "zzz_rare" in on and "aaa_common" in on
    assert on.index("zzz_rare") < on.index("aaa_common")
    assert off.index("aaa_common") < off.index("zzz_rare")


# --- the active-belief count is the lane's whole cost ---------------------


def test_active_count_is_not_recomputed_per_query(store, monkeypatch) -> None:
    """`count_active_beliefs()` scans every row — no index covers a bare
    count on `valid_to IS NULL` — and at 1.125 ms it dominated the lane,
    making it ~25x slower than the overlap lane it replaces. It is
    memoised on `store_generation()`. Reverting the memo turns this red.
    """
    calls: list[int] = []
    real = store.count_active_beliefs

    def counting() -> int:
        calls.append(1)
        return real()

    monkeypatch.setattr(store, "count_active_beliefs", counting)
    for _ in range(5):
        _order(store, fan=True)
    assert len(calls) == 1, f"recomputed {len(calls)} times across 5 queries"


def test_a_write_invalidates_the_memo(store, monkeypatch) -> None:
    """Exact, not merely fresh: `store_generation()` is bumped in the same
    transaction as every content mutation, so a write that changes the
    count also changes the memo key. A time-based or never-invalidated
    cache would pin a stale N here."""
    before = store._active_belief_count_for_fan()
    store.insert_belief(_mk("new_belief", "a brand new belief"))
    after = store._active_belief_count_for_fan()
    assert after == before + 1


def test_memoising_n_does_not_change_the_ordering(store) -> None:
    """Control. The memo is a performance change only — the cached and
    uncached paths must rank identically, or the fix traded correctness
    for latency."""
    cached = _order(store, fan=True)
    store._active_count_memo = None
    uncached = _order(store, fan=True)
    assert cached == uncached


def test_generation_zero_declines_the_memo(store, monkeypatch) -> None:
    """A pre-v4.2 store reads generation 0 forever, and that is
    indistinguishable from an unmutated one. Caching under an unmoving key
    would pin a stale N for the life of the process, and N is multiplied
    by each belief's overlap count, so a stale value does move the
    ranking. Those stores pay the scan instead.
    """
    monkeypatch.setattr(store, "store_generation", lambda: 0)
    calls: list[int] = []
    real = store.count_active_beliefs

    def counting() -> int:
        calls.append(1)
        return real()

    monkeypatch.setattr(store, "count_active_beliefs", counting)
    for _ in range(3):
        store._active_belief_count_for_fan()
    assert len(calls) == 3, "generation 0 must not be memoised"
    assert store._active_count_memo is None
