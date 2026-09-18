"""#1546 K2 — the mutation guard on the discriminability bound itself.

`scripts/budget_discriminability_census.py` reports EC = D/N and calls it an
exact ceiling on effect. That claim rests on one premise: a budget only ever
truncates a pack, so a pool priced below both arms' budgets renders
byte-identical output and no downstream metric can move.

Agreeing rows on unmodified code prove nothing about that premise. A census
that cannot report a violation is not a check, it is a print statement. So the
load-bearing test here drives the census against a deliberately
budget-sensitive fake packer and requires it to **report a violation**. Until
that test passes, the bound is unverified and every number the census prints
inherits the caveat.

The other tests pin the four structural facts the premise is built from,
including the one a prior design pass got wrong: `retrieval._l25_hits` does
receive a budget-derived cap, so "the budget reaches neither lane" is half
false, and the census varies the L2.5 sub-cap because of it.
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
    and `relevance_budget` is computed from the budget. The trim is still a
    tail truncation, so the monotonicity argument survives — but a grid that
    held the sub-cap fixed would be measuring the floor.
    """
    import inspect

    params = set(inspect.signature(retrieval._l25_hits).parameters)
    assert "l25_token_subbudget" in params
    src = inspect.getsource(retrieval.retrieve_with_tiers)
    assert "effective_l25_subbudget" in src
    assert "min(l25_token_subbudget, l25_room)" in src


def test_pack_with_clusters_returns_the_whole_pool_above_its_price() -> None:
    """Stage 2 skips an over-budget belief with `continue`, not `break`.

    So a budget at or above the pool's total price returns the whole pool, and
    the control below shows the same call returning less when the budget is
    under it — a `0 differences` result on the first arm alone would be
    consistent with a packer that ignores its input.
    """
    beliefs = [_belief(f"b{i}", f"content number {i} " * 4) for i in range(6)]
    by_id = {b.id: b for b in beliefs}
    clusters = [
        RetrievalCluster(
            cluster_id=0,
            member_ids=[b.id for b in beliefs],
            representative_id=beliefs[0].id,
            seed_score=1.0,
        )
    ]
    total = sum(retrieval._belief_tokens(b) for b in beliefs)

    whole = pack_with_clusters(clusters, by_id, token_budget=total)
    assert {b.id for b in whole} == set(by_id)

    starved = pack_with_clusters(clusters, by_id, token_budget=total // 3)
    assert len(starved) < len(whole)


def test_admission_is_monotone_in_budget(census: Any) -> None:
    """Raising a budget never evicts a belief a lower budget admitted.

    This is why the decision rule refuses to license a raise: the answer is a
    property of the mechanism, known before any run.
    """
    store = MemoryStore(":memory:")
    try:
        for i in range(12):
            store.insert_belief(
                _belief(
                    f"mono_{i}",
                    f"tomato plants staked knee height variant {i} "
                    + "padding " * 20,
                )
            )
        previous: set[str] = set()
        for budget in (50, 100, 200, 400, 800, 1600, 3200):
            hits = retrieval.retrieve(
                store, "tomato plants staked", token_budget=budget
            )
            ids = {b.id for b in hits}
            assert previous <= ids, (budget, sorted(previous - ids))
            previous = ids
        assert previous, "the fixture retrieved nothing at any budget"
    finally:
        store.close()


# --- The control, and then K2 -----------------------------------------


def test_the_census_reports_no_violation_on_shipped_code(census: Any) -> None:
    """The control arm. On its own this proves nothing; see K2 below."""
    rep = census.report()
    assert rep["violations"] == []
    assert rep["n"] > 0


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
