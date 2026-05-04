"""Three phantom-generation strategies for the #228 bake-off.

Per ``docs/v2_wonder_consolidation.md`` §"The three candidate strategies":

* **RW (random walk)** — start from a high-uncertainty atom, walk N
  hops on the typed-edge graph, bundle visited atoms.
* **TC (triangle closure)** — find pairs (A, B) where A→C and B→C
  exist with edge type ∈ {SUPPORTS, CITES, RELATES_TO}; propose
  (A, B) keyed on shared target C.
* **STS (span-topic sampling)** — sample compositions whose
  constituents span sessions (max session-id diversity in the
  no-embedding form).

Each strategy is a pure function over a ``MemoryStore``. None
mutate the store. Output ordering is sorted by composition for
determinism — the Jaccard evaluator and seed-sweep depend on it.
"""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from aelfrice.models import (
    EDGE_CITES,
    EDGE_RELATES_TO,
    EDGE_SUPPORTS,
)
from aelfrice.scoring import uncertainty_score

from .models import (
    STRATEGY_RW,
    STRATEGY_STS,
    STRATEGY_TC,
    Phantom,
)

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

# Edge types eligible for TC's shared-target shape. Mirrors the
# spec § "The three candidate strategies" gloss for TC. CONTRADICTS
# and SUPERSEDES are excluded because two beliefs sharing a
# CONTRADICTS target are not candidates for composition — they're
# candidates for resolution, which is a different surface.
TC_EDGE_TYPES: frozenset[str] = frozenset({
    EDGE_SUPPORTS,
    EDGE_CITES,
    EDGE_RELATES_TO,
})

# Default RW seed-selection floor. Beta(1,1) has differential entropy
# 0; beliefs with α≈β≈1 (uninformative prior, the synthetic-corpus
# default) sit just above. Tighter beliefs slip below 0; flatter ones
# climb above. The ratification's "0.7" referred to a normalized
# scale not used in v1.x; the floor here is differential-entropy
# units. Tunable per-bake-off via the ``uncertainty_floor`` arg.
DEFAULT_RW_UNCERTAINTY_FLOOR: float = -0.5


def _all_belief_ids(store: "MemoryStore") -> list[str]:
    return store.list_belief_ids()


def random_walk(
    store: "MemoryStore",
    *,
    rng: random.Random,
    n_walks: int = 50,
    depth: int = 2,
    uncertainty_floor: float = DEFAULT_RW_UNCERTAINTY_FLOOR,
) -> list[Phantom]:
    """RW: start from high-uncertainty atoms; walk ``depth`` hops.

    Seed selection: any belief whose ``uncertainty_score(α, β)``
    exceeds ``uncertainty_floor`` is eligible. RNG samples
    ``n_walks`` seeds *with replacement* — empirically observed
    sample variance is what the spec wants to falsify, so resampling
    is allowed.

    Walk: at each step, pick a uniformly random outgoing edge and
    follow it. Self-revisits are dropped from the bundle but counted
    in cost (atoms touched). A walk that hits a dead-end before
    completing ``depth`` hops still produces a phantom from the
    truncated bundle, as long as it has ≥2 atoms.
    """
    eligible: list[str] = []
    for bid in _all_belief_ids(store):
        b = store.get_belief(bid)
        if b is None:
            continue
        if uncertainty_score(b.alpha, b.beta) >= uncertainty_floor:
            eligible.append(bid)
    if not eligible:
        return []

    phantoms: list[Phantom] = []
    for _ in range(n_walks):
        seed = rng.choice(eligible)
        visited: list[str] = [seed]
        seen: set[str] = {seed}
        cost = 1.0  # the seed itself
        cursor = seed
        for _ in range(depth):
            outgoing = store.edges_from(cursor)
            if not outgoing:
                break
            edge = rng.choice(outgoing)
            cost += 1.0
            cursor = edge.dst
            if cursor not in seen:
                seen.add(cursor)
                visited.append(cursor)
        if len(visited) < 2:
            continue
        phantoms.append(
            Phantom(
                composition=tuple(sorted(visited)),
                strategy=STRATEGY_RW,
                construction_cost=cost,
                seed_id=seed,
            )
        )
    # Deduplicate by composition; first occurrence wins on cost.
    seen_comp: dict[tuple[str, ...], Phantom] = {}
    for p in phantoms:
        if p.composition not in seen_comp:
            seen_comp[p.composition] = p
    return sorted(seen_comp.values(), key=lambda p: p.composition)


def triangle_closure(
    store: "MemoryStore",
    *,
    edge_types: frozenset[str] = TC_EDGE_TYPES,
) -> list[Phantom]:
    """TC: propose (A, B) for every pair with a shared edge target C.

    Deterministic — no RNG. Cost per phantom is 3 (A + B + C touched).
    Skips self-pairs and dedup'd unordered pairs (the resulting
    composition tuple is sorted, so duplicates collapse naturally).
    """
    # target_id -> set of source ids that point at it via an eligible
    # edge type.
    incoming: dict[str, set[str]] = {}
    for edge in store.iter_all_edges():
        if edge.type not in edge_types:
            continue
        if edge.src == edge.dst:
            continue
        incoming.setdefault(edge.dst, set()).add(edge.src)

    phantoms: dict[tuple[str, ...], Phantom] = {}
    for target, sources in incoming.items():
        if len(sources) < 2:
            continue
        ordered = sorted(sources)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                comp = (a, b) if a < b else (b, a)
                if comp in phantoms:
                    continue
                phantoms[comp] = Phantom(
                    composition=comp,
                    strategy=STRATEGY_TC,
                    construction_cost=3.0,
                    seed_id=None,
                )
    return sorted(phantoms.values(), key=lambda p: p.composition)


def span_topic_sampling(
    store: "MemoryStore",
    *,
    rng: random.Random,
    n_samples: int = 50,
    composition_size: int = 2,
) -> list[Phantom]:
    """STS: sample compositions that span the most distinct sessions.

    No-embedding form per the spec: session-id diversity stands in
    for "topic" diversity. Beliefs without a ``session_id`` are
    bucketed under a synthetic ``"__none__"`` session — that bucket
    contributes no diversity, so phantoms drawn entirely from it are
    discarded.

    Sampling: pick ``composition_size`` distinct sessions uniformly
    at random, then one belief uniformly from each. Repeat
    ``n_samples`` times. Cost per phantom is ``composition_size``
    (one atom touched per slot).
    """
    if composition_size < 2:
        raise ValueError("composition_size must be >= 2")
    by_session: dict[str, list[str]] = {}
    for bid in _all_belief_ids(store):
        b = store.get_belief(bid)
        if b is None:
            continue
        key = b.session_id if b.session_id else "__none__"
        by_session.setdefault(key, []).append(bid)
    real_sessions = [k for k in by_session if k != "__none__"]
    if len(real_sessions) < composition_size:
        return []
    real_sessions.sort()  # deterministic seeding into the rng

    phantoms: dict[tuple[str, ...], Phantom] = {}
    for _ in range(n_samples):
        chosen_sessions = rng.sample(real_sessions, composition_size)
        picks: list[str] = []
        for sess in chosen_sessions:
            picks.append(rng.choice(sorted(by_session[sess])))
        if len(set(picks)) < composition_size:
            continue
        comp = tuple(sorted(picks))
        if comp in phantoms:
            continue
        phantoms[comp] = Phantom(
            composition=comp,
            strategy=STRATEGY_STS,
            construction_cost=float(composition_size),
            seed_id=None,
        )
    return sorted(phantoms.values(), key=lambda p: p.composition)


__all__ = [
    "DEFAULT_RW_UNCERTAINTY_FLOOR",
    "TC_EDGE_TYPES",
    "random_walk",
    "span_topic_sampling",
    "triangle_closure",
]
