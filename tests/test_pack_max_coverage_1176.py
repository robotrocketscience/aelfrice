"""Budgeted maximum coverage pack selector (#1176 proposal 2).

The load-bearing test is `test_celf_is_byte_identical_to_eager_greedy`: CELF
is a lazy evaluation strategy, not an approximation, so the two must agree on
every input. The proposal names that equality as its own unit test.
"""
from __future__ import annotations

import random

import pytest

from aelfrice.clustering import pack_max_coverage
from aelfrice.models import BELIEF_FACTUAL, LOCK_NONE, Belief

TERMS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]


def _b(bid: str, content: str = "x") -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-07-31T00:00:00Z",
        last_retrieved_at=None,
    )


def _eager_greedy(
    candidates, *, token_budget, coverage, term_weights, cost_fn
):
    """Reference implementation: recompute every marginal, every round.

    Deliberately the naive O(K * n) form. If CELF ever disagrees with this,
    CELF is wrong -- the lazy bounds are only valid under submodularity, and
    a bug in the heap bookkeeping shows up here and nowhere else.
    """
    n = len(candidates)
    rank_of = {b.id: i for i, b in enumerate(candidates)}
    rank_weight = {b.id: (n - i) / n for i, b in enumerate(candidates)}

    def gain(b, covered):
        new = coverage.get(b.id, frozenset()) - covered
        return sum(term_weights.get(t, 0.0) for t in new) * rank_weight[b.id]

    covered = frozenset()
    chosen = []
    used = 0
    remaining = list(candidates)
    while True:
        best = None
        best_key = None
        for b in remaining:
            c = cost_fn(b)
            if used + c > token_budget:
                continue
            g = gain(b, covered)
            ratio = g / c if c > 0 else g
            key = (-ratio, rank_of[b.id], b.id)
            if best_key is None or key < best_key:
                best, best_key = b, key
        if best is None:
            break
        chosen.append(best)
        covered |= coverage.get(best.id, frozenset())
        used += cost_fn(best)
        remaining.remove(best)

    def f(sel):
        u = frozenset()
        for b in sel:
            u |= coverage.get(b.id, frozenset())
        return sum(term_weights.get(t, 0.0) for t in u)

    best_single, best_val = [], -1.0
    for b in candidates:
        if cost_fn(b) > token_budget:
            continue
        v = sum(term_weights.get(t, 0.0) for t in coverage.get(b.id, frozenset()))
        if v > best_val:
            best_val, best_single = v, [b]
    winner = chosen if f(chosen) >= best_val else best_single
    return sorted(winner, key=lambda b: rank_of[b.id])


def _random_case(rng):
    n = rng.randint(1, 14)
    cands = [_b(f"b{i:02d}") for i in range(n)]
    coverage = {
        b.id: frozenset(rng.sample(TERMS, rng.randint(0, len(TERMS))))
        for b in cands
    }
    weights = {t: round(rng.uniform(0.5, 12.0), 3) for t in TERMS}
    costs = {b.id: rng.randint(1, 60) for b in cands}
    budget = rng.randint(0, 200)
    return cands, coverage, weights, (lambda b: costs[b.id]), budget


@pytest.mark.parametrize("seed", range(60))
def test_celf_is_byte_identical_to_eager_greedy(seed: int) -> None:
    """CELF changes which marginals are evaluated, never which are chosen."""
    rng = random.Random(seed)
    cands, cov, w, cost, budget = _random_case(rng)
    got = pack_max_coverage(
        cands, token_budget=budget, coverage=cov, term_weights=w, cost_fn=cost
    )
    want = _eager_greedy(
        cands, token_budget=budget, coverage=cov, term_weights=w, cost_fn=cost
    )
    assert [b.id for b in got] == [b.id for b in want]


def test_the_reference_and_the_implementation_are_not_the_same_code() -> None:
    """Guard against the equality test above going vacuous.

    If someone 'simplifies' the reference to call the implementation, the
    parametrized equality test becomes a tautology that passes against any
    bug. Assert the reference stands alone.
    """
    import inspect

    src = inspect.getsource(_eager_greedy)
    assert "pack_max_coverage" not in src


def test_redundant_belief_is_skipped_for_a_novel_one() -> None:
    """The point of the objective: a restatement earns no marginal gain.

    b0 and b1 cover the same terms; b2 covers a different one. At a budget
    that fits exactly two, the pack must be {b0, b2}, not {b0, b1} -- which
    is what a rank-greedy fill returns.
    """
    cands = [_b("b0"), _b("b1"), _b("b2")]
    cov = {
        "b0": frozenset({"alpha", "beta"}),
        "b1": frozenset({"alpha", "beta"}),
        "b2": frozenset({"gamma"}),
    }
    w = {"alpha": 5.0, "beta": 5.0, "gamma": 4.0}
    out = pack_max_coverage(
        cands, token_budget=20, coverage=cov, term_weights=w,
        cost_fn=lambda b: 10,
    )
    assert [b.id for b in out] == ["b0", "b2"]


def test_rank_order_still_breaks_ties_between_equal_coverage() -> None:
    """With nothing to separate them on coverage, rank decides."""
    cands = [_b("b0"), _b("b1")]
    cov = {"b0": frozenset({"alpha"}), "b1": frozenset({"alpha"})}
    out = pack_max_coverage(
        cands, token_budget=10, coverage=cov, term_weights={"alpha": 3.0},
        cost_fn=lambda b: 10,
    )
    assert [b.id for b in out] == ["b0"]


def test_best_single_arm_wins_when_greedy_is_cost_myopic() -> None:
    """The arm that carries the (1 - 1/e) bound.

    The greedy takes cheap high-ratio crumbs and then cannot afford the one
    belief covering nearly everything. Dropping the single-element arm is
    exactly how the approximation guarantee is lost, so the case is pinned.
    """
    cands = [_b("cheap"), _b("whale")]
    cov = {
        "cheap": frozenset({"alpha"}),
        "whale": frozenset({"beta", "gamma", "delta", "epsilon"}),
    }
    w = {"alpha": 1.0, "beta": 9.0, "gamma": 9.0, "delta": 9.0, "epsilon": 9.0}
    costs = {"cheap": 1, "whale": 100}
    out = pack_max_coverage(
        cands, token_budget=100, coverage=cov, term_weights=w,
        cost_fn=lambda b: costs[b.id],
    )
    assert [b.id for b in out] == ["whale"]


def test_budget_is_never_exceeded() -> None:
    rng = random.Random(99)
    for _ in range(40):
        cands, cov, w, cost, budget = _random_case(rng)
        out = pack_max_coverage(
            cands, token_budget=budget, coverage=cov, term_weights=w,
            cost_fn=cost,
        )
        assert sum(cost(b) for b in out) <= budget


def test_output_is_in_rerank_order_not_selection_order() -> None:
    """Downstream consumes the pack as a ranked list."""
    cands = [_b(f"b{i}") for i in range(4)]
    cov = {
        "b0": frozenset({"alpha"}),
        "b1": frozenset({"beta"}),
        "b2": frozenset({"gamma", "delta"}),
        "b3": frozenset({"epsilon"}),
    }
    w = {t: 5.0 for t in TERMS}
    out = pack_max_coverage(
        cands, token_budget=40, coverage=cov, term_weights=w,
        cost_fn=lambda b: 10,
    )
    ids = [b.id for b in out]
    assert ids == sorted(ids)


def test_determinism_is_not_inherited_from_dict_order() -> None:
    """Same inputs, shuffled insertion order -> identical pack.

    Set and dict iteration order is the classic way a 'deterministic'
    selector stops being one.
    """
    rng = random.Random(7)
    cands, cov, w, cost, budget = _random_case(rng)
    base = pack_max_coverage(
        cands, token_budget=budget, coverage=cov, term_weights=w, cost_fn=cost
    )
    for _ in range(5):
        items = list(cov.items())
        rng.shuffle(items)
        shuffled_cov = dict(items)
        witems = list(w.items())
        rng.shuffle(witems)
        again = pack_max_coverage(
            cands, token_budget=budget, coverage=shuffled_cov,
            term_weights=dict(witems), cost_fn=cost,
        )
        assert [b.id for b in again] == [b.id for b in base]


def test_belief_absent_from_coverage_contributes_nothing() -> None:
    """A belief that matched on another lane covers no query term.

    It must not be treated as covering everything or crash the lookup.
    """
    cands = [_b("known"), _b("unknown")]
    cov = {"known": frozenset({"alpha"})}
    out = pack_max_coverage(
        cands, token_budget=10, coverage=cov, term_weights={"alpha": 2.0},
        cost_fn=lambda b: 10,
    )
    assert [b.id for b in out] == ["known"]


@pytest.mark.parametrize(
    ("cands", "budget"),
    [([], 100), ([_b("b0")], 0), ([_b("b0")], -5)],
)
def test_degenerate_inputs_return_empty(cands, budget) -> None:
    assert pack_max_coverage(
        cands, token_budget=budget, coverage={}, term_weights={},
        cost_fn=lambda b: 1,
    ) == []


def test_zero_cost_belief_does_not_divide_by_zero() -> None:
    cands = [_b("free"), _b("paid")]
    cov = {"free": frozenset({"alpha"}), "paid": frozenset({"beta"})}
    costs = {"free": 0, "paid": 5}
    out = pack_max_coverage(
        cands, token_budget=5, coverage=cov,
        term_weights={"alpha": 1.0, "beta": 1.0},
        cost_fn=lambda b: costs[b.id],
    )
    assert [b.id for b in out] == ["free", "paid"]


# --- retrieval wiring ------------------------------------------------------


from aelfrice.retrieval import retrieve_v2  # noqa: E402


class TestRetrievalWiring:
    """The selector must be reachable from `retrieve_v2` and inert until asked."""

    @staticmethod
    def _store(tmp_path):
        from aelfrice.store import MemoryStore

        s = MemoryStore(str(tmp_path / "p.db"))
        # Every belief matches the query, so all four are L1 candidates and
        # the pack has to choose. b0 and b1 cover the same two query terms;
        # b2 and b3 each cover a term nothing else does. A rank fill takes
        # b0 then b1 (redundant); coverage should reach for b2 / b3.
        s.insert_belief(_b("b0", "retrieval budget retrieval budget retrieval"))
        s.insert_belief(_b("b1", "retrieval budget budget retrieval budget"))
        s.insert_belief(_b("b2", "retrieval locks locks locks locks locks"))
        s.insert_belief(_b("b3", "budget decay decay decay decay decay"))
        return s

    def test_flag_defaults_off_and_pack_is_unchanged(
        self, tmp_path, monkeypatch
    ) -> None:
        from aelfrice.retrieval import is_max_coverage_pack_enabled, retrieve_v2

        monkeypatch.delenv("AELFRICE_MAX_COVERAGE_PACK", raising=False)
        assert is_max_coverage_pack_enabled() is False
        s = self._store(tmp_path)
        base = [b.id for b in retrieve_v2(s, "retrieval budget", budget=4000).beliefs]
        assert base, "fixture retrieved nothing — the test would be vacuous"
        again = [b.id for b in retrieve_v2(s, "retrieval budget", budget=4000).beliefs]
        assert again == base
        s.close()

    def test_env_flag_turns_the_lane_on(self, tmp_path, monkeypatch) -> None:
        from aelfrice.retrieval import is_max_coverage_pack_enabled

        monkeypatch.setenv("AELFRICE_MAX_COVERAGE_PACK", "1")
        assert is_max_coverage_pack_enabled() is True
        monkeypatch.setenv("AELFRICE_MAX_COVERAGE_PACK", "0")
        assert is_max_coverage_pack_enabled() is False

    def test_the_lane_is_actually_reached_when_the_flag_is_on(
        self, tmp_path, monkeypatch
    ) -> None:
        """Wiring assert: `pack_max_coverage` is called iff the flag is on.

        Asserted by observing the call rather than by diffing the pack. A
        retrieval-level output difference is genuinely hard to construct on
        a small fixture, and the reason is the stage-2 finding itself:
        BM25's idf weighting already pushes term-diverse beliefs to the top,
        so the coverage objective and a rank fill agree on most inputs. A
        fixture contorted until they disagreed would assert something the
        real corpus does not do.
        """
        from aelfrice import retrieval as _r

        calls: list[int] = []
        real = _r.pack_max_coverage

        def spy(candidates, **kw):
            calls.append(len(candidates))
            return real(candidates, **kw)

        monkeypatch.setattr(_r, "pack_max_coverage", spy)
        s = self._store(tmp_path)
        q = "retrieval budget locks decay"

        monkeypatch.delenv("AELFRICE_MAX_COVERAGE_PACK", raising=False)
        off = [b.id for b in retrieve_v2(s, q, budget=24).beliefs]
        assert calls == [], "lane ran with the flag off"

        monkeypatch.setenv("AELFRICE_MAX_COVERAGE_PACK", "1")
        on = [b.id for b in retrieve_v2(s, q, budget=24).beliefs]
        s.close()
        assert calls, "lane did not run with the flag on"
        assert off, "fixture retrieved nothing — the test would be vacuous"
        assert on, "coverage pack was empty"

    def test_lane_off_is_byte_identical_to_main(
        self, tmp_path, monkeypatch
    ) -> None:
        """Default path must not move. The flag ships off."""
        s = self._store(tmp_path)
        q = "retrieval budget locks decay"
        monkeypatch.delenv("AELFRICE_MAX_COVERAGE_PACK", raising=False)
        a = [b.id for b in retrieve_v2(s, q, budget=24).beliefs]
        monkeypatch.setenv("AELFRICE_MAX_COVERAGE_PACK", "0")
        b = [b.id for b in retrieve_v2(s, q, budget=24).beliefs]
        s.close()
        assert a == b and a

    def test_coverage_inputs_degrade_without_an_index(self, tmp_path) -> None:
        """No BM25F index -> uniform weights, not an empty objective."""
        from aelfrice.retrieval import _coverage_inputs

        cands = [_b("b0", "alpha beta"), _b("b1", "gamma")]
        cov, w = _coverage_inputs("alpha gamma", cands, None)
        assert cov["b0"] and cov["b1"]
        assert set(w.values()) == {1.0}

    def test_coverage_inputs_are_empty_for_an_empty_query(self) -> None:
        from aelfrice.retrieval import _coverage_inputs

        cov, w = _coverage_inputs("   ", [_b("b0", "alpha")], None)
        assert cov == {} and w == {}
