"""#1546 K2 — the mutation guard on the discriminability bound itself.

`scripts/budget_discriminability_census.py` reports EC = D/N and calls it an
exact ceiling on effect. That claim rests on one **conditional** premise: a
pool priced below both arms' budgets renders byte-identical output, so no
downstream metric can move. It does not rest on the budget being a truncation,
and the tests below show why the difference matters —
`clustering.pack_with_clusters` skips an over-budget belief and continues, so
the budget selects rather than truncates and raising it can evict a belief a
lower budget admitted.

Agreeing rows on unmodified code prove nothing about the premise. A census that
cannot report a violation is not a check, it is a print statement. So the
load-bearing tests here drive the census against three deliberately
budget-sensitive fake packers — one that truncates by item count, one that
returns less at a higher budget, one whose budget moves only a non-gold belief
— and require it to **report a violation** for each. Until they pass, the bound
is unverified and every number the census prints inherits the caveat.

The other tests pin the structural facts the premise is built from, including
the one a prior design pass got wrong: `retrieval._l25_hits` does receive a
budget-derived cap, so "the budget reaches neither lane" is half false, and the
census varies the L2.5 sub-cap because of it. Every fixture prices its beliefs
unequally: under equal prices the greedy fill is a prefix, and a prefix makes a
truncating packer and the shipped one indistinguishable.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from aelfrice import retrieval
from aelfrice.clustering import RetrievalCluster, pack_with_clusters
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_UNKNOWN,
    Belief,
)
from aelfrice.store import MemoryStore

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "budget_discriminability_census.py"
)


def _load_census() -> Any:
    """Import the census as a module, the `test_migration_policy.py` way."""
    spec = importlib.util.spec_from_file_location(
        "budget_discriminability_census", str(_SCRIPT_PATH)
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["budget_discriminability_census"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def census() -> Any:
    return _load_census()


def _belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"t_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_UNKNOWN,
    )


# --- The premise, checked rather than read ----------------------------


def test_l1_hits_takes_no_budget_parameter() -> None:
    """Candidate generation does not vary with the budget.

    Asserted on the signature rather than on behaviour, because the claim is
    that no budget can reach the lane at all: a behavioural check over some
    budgets cannot distinguish "does not read it" from "did not bind here".
    """
    import inspect

    params = set(inspect.signature(retrieval._l1_hits).parameters)
    assert "token_budget" not in params
    assert not any("budget" in p for p in params), sorted(params)


def test_l25_hits_does_take_a_budget_derived_cap() -> None:
    """The half of the prior claim that is false, pinned so it stays visible.

    A prior design pass concluded by code reading that `token_budget` reaches
    neither `_l1_hits` nor `_l25_hits`. It reaches `_l25_hits`:
    `retrieve_with_tiers` passes `min(l25_token_subbudget, relevance_budget)`,
    and `relevance_budget` is computed from the budget. A grid that held the
    sub-cap fixed would be measuring the floor rather than the budget.
    """
    import inspect

    params = set(inspect.signature(retrieval._l25_hits).parameters)
    assert "l25_token_subbudget" in params
    src = inspect.getsource(retrieval.retrieve_with_tiers)
    assert "effective_l25_subbudget" in src
    assert "min(l25_token_subbudget, l25_room)" in src


def _unequal_cost_pack_fixture() -> tuple[
    list[RetrievalCluster], dict[str, Belief]
]:
    """A pool whose beliefs have unequal prices, with the dear one first.

    Equal prices make the greedy a prefix, and a prefix hides the whole
    behaviour of a skip-and-continue fill: every budget returns the first k
    items for some k, so a truncating packer and the shipped one are
    indistinguishable. Every fixture here is therefore deliberately unequal.
    """
    beliefs = [
        _belief("dear", "expensive belief " * 12),
        _belief("cheap_a", "short a"),
        _belief("cheap_b", "short b"),
        _belief("cheap_c", "short c"),
    ]
    by_id = {b.id: b for b in beliefs}
    clusters = [
        RetrievalCluster(
            cluster_id=0,
            member_ids=[b.id for b in beliefs],
            representative_id=beliefs[0].id,
            seed_score=1.0,
        )
    ]
    return clusters, by_id


def test_pack_with_clusters_returns_the_whole_pool_above_its_price() -> None:
    """A budget at or above the pool's total price returns the whole pool.

    This is the bound the census rests on, at the packer. The control below
    shows the same call returning less when the budget is under the price — a
    `0 differences` result on the first arm alone would be consistent with a
    packer that ignores its input.
    """
    clusters, by_id = _unequal_cost_pack_fixture()
    costs = {b.id: retrieval._belief_tokens(b) for b in by_id.values()}
    assert len(set(costs.values())) > 1, costs
    total = sum(costs.values())

    whole = pack_with_clusters(clusters, by_id, token_budget=total)
    assert {b.id for b in whole} == set(by_id)

    starved = pack_with_clusters(clusters, by_id, token_budget=total // 3)
    assert len(starved) < len(whole)


def test_pack_with_clusters_skips_an_over_budget_belief_and_continues() -> None:
    """A skipped belief does not end the fill, so the budget *selects*.

    At a budget that cannot afford `dear` but can afford the cheap tail, the
    pack returns the cheap tail rather than stopping at the first item that
    does not fit. The fixture puts all four beliefs in one cluster on purpose:
    stage 2's `continue` skips within a cluster, and its outer loop moves on to
    the next cluster regardless, so a pool of singleton clusters skips either
    way and cannot tell the two apart.
    """
    clusters, by_id = _unequal_cost_pack_fixture()
    costs = {b.id: retrieval._belief_tokens(b) for b in by_id.values()}
    budget = costs["dear"] - 1
    assert budget >= costs["cheap_a"] + costs["cheap_b"], costs

    out = [b.id for b in pack_with_clusters(clusters, by_id, token_budget=budget)]
    assert "dear" not in out
    assert out, "the pack stopped at the first belief that did not fit"


def test_raising_a_budget_can_evict_a_belief_a_lower_budget_admitted() -> None:
    """Admission is **not** monotone in budget, on the shipped retrieve path.

    The pre-registration originally asserted the opposite and rested its
    licensing logic on it. It is false: `pack_with_clusters` skips what it
    cannot afford and keeps filling — stage 1 abandons on the first
    over-budget representative and stage 2 then walks the rest — so the budget
    selects rather than truncates. A budget that cannot afford a dear
    high-ranked belief spends itself on cheaper lower-ranked ones, and raising
    it by one token evicts them.

    The fixture prices its beliefs unequally on purpose. Under the equal-cost
    fixture this test used to carry, the greedy is a prefix and the assertion
    cannot fail on any packer, truncating or not.
    """
    query = "tomato staked"
    store = MemoryStore(":memory:")
    try:
        dear = _belief(
            "dear", "tomato staked " * 12 + "gardening notes for the season"
        )
        lean = _belief("lean", "tomato staked lightly")
        for b in (dear, lean):
            store.insert_belief(b)
        costs = {b.id: retrieval._belief_tokens(b) for b in (dear, lean)}
        assert len(set(costs.values())) > 1, costs

        pool = retrieval.retrieve(store, query, token_budget=10**9)
        assert [b.id for b in pool] == ["dear", "lean"], [b.id for b in pool]

        evictions: list[tuple[int, int, list[str]]] = []
        previous: set[str] = set()
        previous_budget = 0
        for budget in range(1, sum(costs.values()) + 2):
            ids = {
                b.id
                for b in retrieval.retrieve(store, query, token_budget=budget)
            }
            lost = sorted(previous - ids)
            if lost:
                evictions.append((previous_budget, budget, lost))
            previous, previous_budget = ids, budget

        assert evictions, (
            "no budget increase evicted anything, so either the packer became "
            "monotone or the fixture no longer prices its beliefs unequally: "
            f"{costs}"
        )
        assert previous == {"dear", "lean"}
    finally:
        store.close()


def test_no_budget_at_or_above_the_pool_price_changes_the_output() -> None:
    """The bound that survives: above the pool's price, every budget agrees.

    This is the conditional claim the census's ceiling actually rests on, and
    it is unaffected by the non-monotonicity above because it quantifies only
    over budgets at which no cap binds. The control checks a budget below the
    price does change the output, so agreement above it is not vacuous.
    """
    query = "tomato staked"
    store = MemoryStore(":memory:")
    try:
        store.insert_belief(
            _belief("dear", "tomato staked " * 12 + "gardening notes")
        )
        store.insert_belief(_belief("lean", "tomato staked lightly"))
        pool = retrieval.retrieve(store, query, token_budget=10**9)
        price = sum(retrieval._belief_tokens(b) for b in pool)
        baseline = [b.id for b in pool]

        for budget in range(price, price * 4, 7):
            ids = [
                b.id
                for b in retrieval.retrieve(store, query, token_budget=budget)
            ]
            assert ids == baseline, (budget, price, ids)

        below = [
            b.id
            for b in retrieval.retrieve(store, query, token_budget=price - 1)
        ]
        assert below != baseline, (price, below)
    finally:
        store.close()


# --- The control, and then K2 -----------------------------------------


def test_the_census_reports_no_violation_on_shipped_code(census: Any) -> None:
    """The control arm. On its own this proves nothing; see K2 below."""
    rep = census.report()
    assert rep["violations"] == []
    assert rep["n"] > 0


@pytest.mark.timeout(120)
def test_containment_is_a_property_of_this_corpus_and_not_a_theorem(
    census: Any,
) -> None:
    """A lower budget CAN reach a belief the unbudgeted probe did not.

    The clean control above invites the inference that the probe at
    `POOL_PROBE_BUDGET` is a structural upper bound on every arm, so the
    containment guard can never fire and needs no fix. It is not, and
    the difference decides whether the guard is doing work: the census
    reports zero because `_open_store` writes **no edges**, not because
    escape is impossible.

    Add one `TEMPORAL_NEXT` edge and the shipped code escapes:

    1. The spine lane seeds from `l1_packed[:DEFAULT_SPINE_SEED_COUNT]`.
    2. `l1_packed` at a low budget is **not a prefix** of `l1_packed` at
       a high one. At the probe budget the dear beliefs are packed and
       hold every seed seat; at a smaller budget the later ones do not
       fit, later stages still add the cheap tail, and a cheap belief
       lands at an earlier position than it held at the probe — inside
       the seed window rather than outside it.

       Note this is broader than the stage-2 `continue` that
       `test_raising_a_budget_can_evict_a_belief_a_lower_budget_admitted`
       pins. Mutating that `continue` to a `break` does **not** stop the
       escape, because the cheap tail is re-added downstream either way.
       So the escape does not rest on the non-monotone skip alone; the
       position shift is enough, and the seed window is what converts a
       position shift into a candidacy change.
    3. That promoted belief's spine neighbour shares no term with the
       query, so it is neither an L1 nor an L2.5 hit and is reachable
       only through the spine — from a seed the probe never had.

    Observed on this fixture, with `ups` at its shipped 1500:

        probe (1e9): d0 d1 d2 d3 d4 c0..c7          — no offlex
        arm    750 : d0 d1       c0..c7 offlex      — escaped
        arm   1125 : d0 d1 d2    c0..c7 offlex      — escaped
        arm   1500 : d0 d1 d2 d3 c0..c7 offlex      — escaped

    This matters beyond bookkeeping: any future corpus whose stores
    carry edges runs the containment check for real, and a run that
    treats a violation as impossible would read a real escape as a bug
    in the guard.

    Load-bearing, checked by mutation: setting `DEFAULT_SPINE_SEED_COUNT`
    to 0 makes this red. Environment controls cannot reach it, because
    the census clears every `AELFRICE_*` variable from its own process.
    """
    from aelfrice.models import EDGE_TEMPORAL_NEXT, EDGE_VALENCE, Edge
    from aelfrice.temporal_spine import DEFAULT_SPINE_SEED_COUNT

    # Exactly enough dear beliefs to fill the seed window at the probe
    # budget, each ranking above the cheap tail on term frequency and
    # each priced so that a small arm can afford only the first two.
    dear = [
        census._belief(f"d{i}", "alpha " * 300)
        for i in range(DEFAULT_SPINE_SEED_COUNT)
    ]
    cheap = [census._belief(f"c{i}", f"alpha cheap {i}") for i in range(8)]
    # Shares no query term, so neither L1 nor L2.5 can reach it.
    off_lexicon = census._belief("offlex", "zzz unrelated neighbour zzz")

    store = MemoryStore(":memory:")
    for b in (*dear, *cheap, off_lexicon):
        store.insert_belief(b)
    # Hang it off the first cheap belief, which sits just outside the
    # seed window at the probe budget and just inside it once the dear
    # beliefs are priced out.
    store.insert_edge(
        Edge(
            src=cheap[0].id,
            dst=off_lexicon.id,
            type=EDGE_TEMPORAL_NEXT,
            weight=EDGE_VALENCE[EDGE_TEMPORAL_NEXT],
        )
    )

    lane = census.lanes()[0]
    probe = {
        b.id for b in census._retrieve(
            store, lane, "alpha",
            token_budget=census.POOL_PROBE_BUDGET,
            l25_token_subbudget=census.POOL_PROBE_BUDGET,
        )
    }
    escaped: set[str] = set()
    for multiplier in census.BUDGET_MULTIPLIERS:
        arm = {
            b.id for b in census._retrieve(
                store, lane, "alpha",
                token_budget=int(lane.budget * multiplier),
                l25_token_subbudget=retrieval.DEFAULT_L25_TOKEN_SUBBUDGET,
            )
        }
        escaped |= arm - probe

    assert escaped, (
        "no arm escaped the unbudgeted probe on an edge-bearing store. "
        "Either the spine seed, the non-monotone packer, or the fixture's "
        "price spread stopped doing its job — re-read this test before "
        "concluding containment is structural"
    )
    assert off_lexicon.id in escaped, escaped


@pytest.mark.timeout(120)
def test_the_bound_fails_against_a_budget_sensitive_packer(
    census: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K2. The census must REPORT A VIOLATION against a fake packer.

    The fake reads the budget as an item count rather than as a price, which is
    exactly the class of packer that would make EC's ceiling claim false: it
    drops beliefs the pool could afford, so two arms whose pools both price
    under their budgets no longer render the same block.

    The assertion is on the census's own violation list, not on a difference
    the test computes itself, because what is under test is the census's
    ability to notice — a check that cannot fail is not a check.
    """

    def fake_pack(
        clusters: list[RetrievalCluster],
        belief_by_id: dict[str, Belief],
        *,
        token_budget: int,
        cluster_diversity_target: int = 3,
        fallback_to_score: bool = True,
        cost_fn: Any = None,
    ) -> list[Belief]:
        ordered: list[Belief] = []
        seen: set[str] = set()
        for cluster in sorted(clusters, key=lambda c: -c.seed_score):
            for mid in cluster.member_ids:
                if mid in seen:
                    continue
                b = belief_by_id.get(mid)
                if b is None:
                    continue
                seen.add(mid)
                ordered.append(b)
        # Budget read as an item count. Nothing about the pool's price is
        # consulted, so the cut lands on pools the budget could afford.
        return ordered[: max(0, token_budget // 400)]

    monkeypatch.setattr(census.retrieval, "pack_with_clusters", fake_pack)
    rep = census.report()
    assert rep["violations"], (
        "the census agreed with a packer that truncates by item count; the "
        "EC ceiling is unverified"
    )
    assert any("pool_cost=" in v for v in rep["violations"])
    assert rep["lanes"]["ups"]["d"] > 0

    # And the same call reports clean once the shipped packer is back, so the
    # violation is attributable to the mutation and not to the fixture.
    monkeypatch.undo()
    assert census.report()["violations"] == []


@pytest.mark.timeout(120)
def test_the_bound_fails_against_a_packer_that_returns_less_at_a_high_budget(
    census: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K2, second arm: the packer class that used to blind the guard.

    The census reads its pool by calling the same `retrieve()` path at
    `POOL_PROBE_BUDGET`, so a packer that returns *less* at a high budget
    shrinks the very pool every footprint is compared against. This fake did
    exactly that and the census reported 0 violations, EC 0 on every lane, and
    exit 0 — while the identical fake confined below the probe budget reported
    252 violations. The containment check is what closes it, so the assertion
    is on the census's own violation list.

    The shipped stage-2 `continue` belongs to this same non-monotone family,
    which is why the arm is not hypothetical.

    The threshold sits above every arm and below the probe deliberately.
    An earlier revision fired at 700, which is below the whole grid of
    three lanes — `ups` runs 750-3000, `retrieval_default` 1200-4800 and
    `rebuilder` 2000-8000 — so on those the fake shrank the probe and
    every arm by the same item, the comparison stayed consistent, and no
    containment violation could arise. The test asserted only that the
    GLOBAL list was non-empty, so the three lanes it could not mutate
    were carried by the three it could, and their containment guard was
    unmutation-tested while reading as covered. The per-lane assertion
    below is what makes that visible.
    """
    shipped = census.retrieval.pack_with_clusters
    # Above `2.0 * budget`, the largest multiplier in BUDGET_MULTIPLIERS,
    # for every lane; far below POOL_PROBE_BUDGET. So the probe is the
    # only call the fake touches, on every lane.
    above_every_arm = 100_000
    assert above_every_arm < census.POOL_PROBE_BUDGET
    arm_budgets = [
        int(lane.budget * m)
        for lane in census.lanes()
        for m in census.BUDGET_MULTIPLIERS
    ]
    assert max(arm_budgets) < above_every_arm, (
        "a lane's grid now reaches the mutation threshold, so the fake "
        f"would shrink that lane's arms too: max arm {max(arm_budgets)}"
    )

    def fake_pack(
        clusters: list[RetrievalCluster],
        belief_by_id: dict[str, Belief],
        **kwargs: Any,
    ) -> list[Belief]:
        out = shipped(clusters, belief_by_id, **kwargs)
        # Returns less the higher the budget goes, the pool probe included.
        if kwargs["token_budget"] >= above_every_arm and out:
            return out[1:]
        return out

    baseline_n = census.report()["n"]
    monkeypatch.setattr(census.retrieval, "pack_with_clusters", fake_pack)
    rep = census.report()
    assert rep["violations"], (
        "the census agreed with a packer that returns less at a higher "
        "budget; its pool probe is measuring itself"
    )
    assert any("unbudgeted probe" in v for v in rep["violations"])
    # Per-lane, not just globally: a lane the fake cannot reach reads as
    # covered when the assertion is on the pooled list.
    lanes_hit = {
        lane.name
        for lane in census.lanes()
        if any(f"{lane.name}/" in v for v in rep["violations"])
    }
    assert lanes_hit == {lane.name for lane in census.lanes()}, (
        "the containment guard is unmutation-tested on "
        f"{sorted({l.name for l in census.lanes()} - lanes_hit)}: the fake "
        "shrinks the probe there but no violation is reported"
    )
    assert rep["n"] == baseline_n, (
        "N moved under the mutation, so queries whose pool the mutation "
        "emptied were counted as measured: N went "
        f"{baseline_n} -> {rep['n']}"
    )

    monkeypatch.undo()
    assert census.report()["violations"] == []


@pytest.mark.timeout(120)
def test_the_bound_fails_when_a_budget_moves_a_non_gold_belief(
    census: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bound is a block-identity claim, so the guard reads the block.

    EC is defined on the gold footprint, and for the *statistic* that is
    right. But the ceiling sentence the census publishes says both arms render
    a byte-identical block. A guard that reads only the gold lines cannot see
    this fake, which drops non-gold beliefs inside a budget window the pool
    probe never visits, and it reported 0 violations on every lane.
    """
    shipped = census.retrieval.pack_with_clusters

    def fake_pack(
        clusters: list[RetrievalCluster],
        belief_by_id: dict[str, Belief],
        **kwargs: Any,
    ) -> list[Belief]:
        out = shipped(clusters, belief_by_id, **kwargs)
        if 700 <= kwargs["token_budget"] <= 10000:
            return [b for b in out if "noise" not in b.id]
        return out

    monkeypatch.setattr(census.retrieval, "pack_with_clusters", fake_pack)
    rep = census.report()
    assert rep["violations"], (
        "the census agreed with a packer whose budget moves a non-gold "
        "belief; its block-identity claim is unchecked"
    )
    assert any("rendered block differs" in v for v in rep["violations"])

    monkeypatch.undo()
    assert census.report()["violations"] == []


def test_an_empty_pool_is_excluded_from_n_rather_than_measured(
    census: Any,
) -> None:
    """A query the probe returns nothing for must not count as measured.

    `degenerate` asks whether the pool is a subset of the gold set, and the
    empty set is. Without the explicit empty-pool exclusion, a query whose pool
    vanished counted in N while contributing nothing to D, which is how a
    mutated run reported a larger N than the shipped one.
    """
    rep = census.report()
    for row in rep["lanes"].values():
        for corpus in row["by_corpus"].values():
            assert "empty_pool_excluded" in corpus
            assert corpus["empty_pool_excluded"] == 0
            assert corpus["pool_size_min"] > 0


def test_the_census_exits_non_zero_when_the_bound_is_violated(
    census: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reported violation must also be a failing exit code.

    A gate that prints `VIOLATION` and exits 0 is the #1160 failure mode: a
    required check reporting green over a red result.
    """
    monkeypatch.setattr(
        census,
        "report",
        lambda: {**_MINIMAL_REPORT, "violations": ["synthetic"]},
    )
    assert census.main([]) == 2
    assert census.main(["--emit-figures"]) == 2


_MINIMAL_REPORT: dict[str, Any] = {
    "instrument": "x",
    "decision_rule": "y",
    "statistic": "z",
    "resolvers": {},
    "grid": {
        "budget_multipliers": [1.0],
        "l25_subbudgets": [400],
        "shipped_cell": [1.0, 400],
        "pool_probe_budget": 1,
    },
    "constants": {},
    "unmeasurable_lanes": {},
    "not_exercised": {},
    "lanes": {
        "search_tool_bash": {
            "symbol": "s",
            "shipped_budget": 300,
            "l1_limit": 5,
            "cost_fn": "c",
            "n": 0,
            "d": 0,
            "ec_pp": None,
            "by_corpus": {
                "benchmark": {
                    "n": 0,
                    "degenerate_excluded": 0,
                    "empty_pool_excluded": 0,
                    "d": 0,
                    "ec_pp": None,
                    "pool_cost_min": 0,
                    "pool_cost_max": 0,
                    "pool_size_min": 0,
                    "pool_size_max": 0,
                    "binds_on": {
                        "pool": 0,
                        "token_budget": 0,
                        "l25_subbudget": 0,
                        "both": 0,
                    },
                    "queries_where_budget_can_bind": [],
                    "whole_corpus": {
                        "whole_store_cost": 0,
                        "beliefs": 0,
                        "largest_store_query": "",
                        "exceeds_shipped_budget": False,
                    },
                }
            },
        }
    },
    "n": 0,
    "grey_band_pp": 9.5,
    "grey_band_has_aa_term": False,
    "required_n_per_arm": 3713,
    "violations": [],
}
