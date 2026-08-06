"""`apply_edge_type_rerank` folds its penalties in a total order (#1375).

The module docstring promises that "the same `(hops, store, penalties)`
input produces byte-identical output". Until #1375 the penalty fold ran
`for edge_type in firing:` over a `set[str]`. Set iteration order for
strings depends on the per-process hash seed, and float multiplication
is not associative — so any belief with two or more distinct firing edge
types got a score whose last bits varied between interpreter runs.

Two tests, because the defect has two faces:

  F1 (`test_penalty_fold_runs_in_sorted_order`) — in-process, and the
     one that reproduces deterministically. One hash seed fixes one set
     layout, so a single fixture would be a coin flip: it reproduces on
     the seeds where that set happens not to iterate in sorted order.
     Sweeping many *distinct* type-name triples inside one process
     removes the coin flip — 64 independent 3-element sets cannot all
     iterate in sorted order, and the test asserts that at least one
     does not before it asserts anything about scores.

  F2 (`test_score_is_byte_identical_across_hash_seeds`) — subprocess,
     and the one that states the shipped contract literally: the same
     input under different `PYTHONHASHSEED` values must produce the same
     bits.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from itertools import permutations
from pathlib import Path

import pytest

from aelfrice.bfs_multihop import ScoredHop
from aelfrice.edge_rerank import apply_edge_type_rerank
from aelfrice.models import BELIEF_FACTUAL, Belief, Edge
from aelfrice.store import MemoryStore

# Three penalties whose product depends on the order they are folded in.
# Chosen by search: 4 of the 5 non-sorted permutations of these give a
# float different from the sorted-order product, so a fold that walks a
# set is caught roughly two times in three per fixture. Round numbers
# like (0.5, 0.25, 0.125) are exact in binary and would make every
# permutation agree — the test would then pass against the bug.
ORDER_SENSITIVE_PENALTIES: tuple[float, float, float] = (0.4, 0.83, 0.94)

# How many distinct type-name triples the in-process sweep uses. Each is
# an independent draw of a set layout; at ~1/3 chance of a triple
# iterating in a product-preserving order, 64 makes a vacuous pass
# (3e-31) impossible in practice rather than merely unlikely.
TRIPLE_COUNT: int = 64

BASE_SCORE: float = 1.0


def _mk(belief_id: str) -> Belief:
    return Belief(
        id=belief_id,
        content=belief_id,
        content_hash=f"h_{belief_id}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level="none",
        locked_at=None,
        created_at="2026-08-05T00:00:00Z",
        last_retrieved_at=None,
    )


def _fold(base: float, values: tuple[float, ...]) -> float:
    out = base
    for v in values:
        out *= v
    return out


def _triple(i: int) -> tuple[str, str, str]:
    """Three synthetic edge-type names whose sorted order is A < B < C.

    Synthetic rather than the `models` edge types because the sweep needs
    many independent set layouts and `models` only declares eleven names.
    `penalties` is an operator-supplied `Mapping[str, float]`, so the
    production path places no constraint on the keys.
    """
    return (f"MARK_{i:02d}_A", f"MARK_{i:02d}_B", f"MARK_{i:02d}_C")


def test_penalty_fold_runs_in_sorted_order() -> None:
    """Every multi-type hop scores exactly as a sorted-order fold.

    Exact `==`, not `approx`: the whole defect lives in the last bits,
    and `approx` would report the bug as a pass.
    """
    penalties: dict[str, float] = {}
    triples = [_triple(i) for i in range(TRIPLE_COUNT)]
    store = MemoryStore(":memory:")
    hops: list[ScoredHop] = []
    for i, names in enumerate(triples):
        for name, value in zip(names, ORDER_SENSITIVE_PENALTIES):
            penalties[name] = value
        target = f"TARGET_{i:02d}"
        store.insert_belief(_mk(target))
        for j, name in enumerate(names):
            src = f"SRC_{i:02d}_{j}"
            store.insert_belief(_mk(src))
            store.insert_edge(Edge(src=src, dst=target, type=name, weight=1.0))
        belief = store.get_belief(target)
        assert belief is not None
        hops.append(
            ScoredHop(belief=belief, score=BASE_SCORE, depth=1, path=["SUPPORTS"])
        )

    # Guard the fixture before trusting the result. If every triple
    # happened to iterate in a product-preserving order under this run's
    # hash seed, the assertions below would hold against the unsorted
    # fold too and would prove nothing.
    exposing = [
        names
        for names in triples
        if _fold(BASE_SCORE, tuple(penalties[n] for n in set(names)))
        != _fold(BASE_SCORE, tuple(penalties[n] for n in sorted(names)))
    ]
    assert exposing, (
        "no triple in this fixture iterates in an order that changes the "
        "product, so the test cannot distinguish a sorted fold from a set "
        "fold — widen TRIPLE_COUNT or repick ORDER_SENSITIVE_PENALTIES"
    )

    result = apply_edge_type_rerank(hops, store, penalties=penalties)
    by_id = {h.belief.id: h.score for h in result}
    assert len(by_id) == TRIPLE_COUNT
    for i, names in enumerate(triples):
        expected = _fold(
            BASE_SCORE, tuple(penalties[n] for n in sorted(names))
        )
        assert by_id[f"TARGET_{i:02d}"] == expected, (
            f"hop {i} folded {names} in a non-sorted order"
        )


def test_order_sensitive_penalties_are_actually_order_sensitive() -> None:
    """The chosen constants must not be associativity-safe.

    A separate test rather than an inline check, because it is the one
    assumption the sweep above cannot repair by adding fixtures: pick
    three exactly-representable factors and every permutation agrees,
    and the sweep silently degrades into a tautology.
    """
    products = {
        _fold(BASE_SCORE, p).hex()
        for p in permutations(ORDER_SENSITIVE_PENALTIES)
    }
    assert len(products) > 1


_DRIVER = '''
import json
from aelfrice.bfs_multihop import ScoredHop
from aelfrice.edge_rerank import apply_edge_type_rerank
from aelfrice.models import BELIEF_FACTUAL, Belief, Edge
from aelfrice.store import MemoryStore

TYPES = {types!r}
PENALTIES = dict(zip(TYPES, {penalties!r}))


def mk(bid):
    return Belief(
        id=bid, content=bid, content_hash="h_" + bid, alpha=1.0, beta=1.0,
        type=BELIEF_FACTUAL, lock_level="none", locked_at=None,
        created_at="2026-08-05T00:00:00Z", last_retrieved_at=None,
    )


store = MemoryStore(":memory:")
store.insert_belief(mk("TARGET"))
for j, t in enumerate(TYPES):
    store.insert_belief(mk("SRC%d" % j))
    store.insert_edge(Edge(src="SRC%d" % j, dst="TARGET", type=t, weight=1.0))
belief = store.get_belief("TARGET")
hop = ScoredHop(belief=belief, score={base!r}, depth=1, path=["SUPPORTS"])
out = apply_edge_type_rerank([hop], store, penalties=PENALTIES)[0]
print(json.dumps({{"score": out.score.hex(), "set_order": list(set(TYPES))}}))
'''

# Enough seeds to see several distinct set layouts, few enough that the
# whole sweep is well under the per-test timeout below (~0.1s per run).
HASH_SEEDS: tuple[str, ...] = ("0", "1", "2", "3", "5", "8", "13", "21")

# Per-driver wall clock. The driver imports `edge_rerank` and builds a
# four-row in-memory store — ~0.1s warm. Generous enough not to flake
# under a loaded CI box, bounded because a test with no exit is not a
# test (`tests/test_test_termination_policy.py`).
DRIVER_TIMEOUT_S: float = 20.0


@pytest.mark.timeout(60)
def test_score_is_byte_identical_across_hash_seeds(tmp_path: Path) -> None:
    """The docstring's byte-identical promise, tested literally."""
    script = tmp_path / "fold_driver.py"
    script.write_text(
        _DRIVER.format(
            types=[f"MARK_{c}" for c in ("A", "B", "C")],
            penalties=list(ORDER_SENSITIVE_PENALTIES),
            base=BASE_SCORE,
        ),
        encoding="utf-8",
    )
    scores: list[str] = []
    orders: set[tuple[str, ...]] = set()
    for seed in HASH_SEEDS:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            env=env,
            check=True,
            timeout=DRIVER_TIMEOUT_S,
        )
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        scores.append(payload["score"])
        orders.add(tuple(payload["set_order"]))

    # Same fixture guard as F1, in the dimension this test varies: if
    # every seed laid the set out identically there would be nothing for
    # a set-ordered fold to get wrong.
    assert len(orders) > 1, (
        f"PYTHONHASHSEED did not change the set layout across "
        f"{len(HASH_SEEDS)} seeds; this test cannot see the defect"
    )
    assert len(set(scores)) == 1, (
        f"score varied with PYTHONHASHSEED: {sorted(set(scores))}"
    )
