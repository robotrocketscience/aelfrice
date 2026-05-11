"""Synthetic corpus + feedback simulator for the #228 bake-off.

Per the spec, R0 is "synthetic 200-atom corpus + feedback simulator
+ evaluator". Synthetic — generated public-side from a seed, not
imported from any private fixture.

Corpus shape:

* ``n_topics`` synthetic topics, each labeled by an integer id.
* ``n_atoms_per_topic`` belief atoms per topic; total
  ``n_topics * n_atoms_per_topic`` ≈ 200 with the defaults.
* Each atom is also stamped with a ``session_id`` drawn from a
  small pool — STS reads this for diversity scoring.
* Per-topic intra-edges seeded so TC has shared-target triangles to
  close. Cross-topic edges seeded sparser so RW has occasional
  bridges but the topic structure stays detectable.
* Each atom carries an α/β prior. Most atoms get the uninformative
  Beta(1,1); a fraction stamped with α=β≈0.7 (high-uncertainty)
  so RW's seed-selection rule fires reliably.

Feedback simulator: given a phantom (composition tuple), decide
``confirm`` if every belief in the composition shares a topic with
the others — the simplest "true relationship" predicate that lets
all three strategies be exercised without one trivially winning.
Otherwise ``junk``. The junk rate that emerges is the corpus's
prior over phantom quality; a strategy beats H0 by producing a
higher confirm fraction than uniform random would.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aelfrice.models import (
    BELIEF_FACTUAL,
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
    CORROBORATION_SOURCE_MCP_REMEMBER,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    EDGE_CITES,
    EDGE_RELATES_TO,
    EDGE_SUPPORTS,
    LOCK_NONE,
    Belief,
    Edge,
)

# Synthetic source-type pool mirrors the four ingest paths that A1
# (#190) plumbs into the `belief_corroborations.source_type` column.
# Order is fixed so multi-seed runs draw the same pool elements.
DEFAULT_SOURCE_TYPE_POOL: tuple[str, ...] = (
    CORROBORATION_SOURCE_COMMIT_INGEST,
    CORROBORATION_SOURCE_TRANSCRIPT_INGEST,
    CORROBORATION_SOURCE_MCP_REMEMBER,
    CORROBORATION_SOURCE_FILESYSTEM_INGEST,
)

if TYPE_CHECKING:
    from aelfrice.store import MemoryStore

# Promotion gate from the substrate decision (#196). A phantom is
# "promoted" once accumulated confirms push α to this threshold.
ALPHA_PROMOTION_THRESHOLD: float = 12.0


@dataclass(frozen=True)
class CorpusAtom:
    belief_id: str
    topic: int
    session_id: str
    source_type: str


@dataclass
class SyntheticCorpus:
    """Ground-truth view of the synthetic corpus the simulator built.

    The bake-off needs both the atoms (to populate a real
    ``MemoryStore``) and the topic mapping (to grade phantoms in
    the feedback simulator). Edges are kept for parity-debugging
    but are not directly read by the evaluator.
    """

    atoms: tuple[CorpusAtom, ...]
    edges: tuple[Edge, ...] = field(default_factory=tuple)

    def topic_of(self, belief_id: str) -> int | None:
        for atom in self.atoms:
            if atom.belief_id == belief_id:
                return atom.topic
        return None

    def session_of(self, belief_id: str) -> str | None:
        for atom in self.atoms:
            if atom.belief_id == belief_id:
                return atom.session_id
        return None

    def source_type_of(self, belief_id: str) -> str | None:
        for atom in self.atoms:
            if atom.belief_id == belief_id:
                return atom.source_type
        return None


def _atom_id(topic: int, idx: int) -> str:
    return f"t{topic:02d}_a{idx:03d}"


def build_corpus(
    *,
    rng: random.Random,
    n_topics: int = 8,
    n_atoms_per_topic: int = 25,
    n_sessions: int = 8,
    n_source_types: int = 4,
    intra_edge_density: float = 0.25,
    cross_edge_density: float = 0.02,
    high_uncertainty_fraction: float = 0.15,
) -> SyntheticCorpus:
    """Generate a deterministic synthetic corpus.

    Density notes (Decision F in the planning memo): the defaults
    were grid-searched on a tiny 3-topic preview to land each
    strategy in its predicted regime. Adjust for sweep variance,
    but the unit tests pin the default shape.
    """
    if n_topics < 2:
        raise ValueError("n_topics must be >= 2 for STS to be testable")
    if n_atoms_per_topic < 3:
        raise ValueError("n_atoms_per_topic must be >= 3 for TC to find triangles")
    if n_sessions < 2:
        raise ValueError("n_sessions must be >= 2 for STS to be testable")
    if n_source_types < 2:
        raise ValueError(
            "n_source_types must be >= 2 for the source-distinct verdict path "
            "to be testable"
        )
    if n_source_types > len(DEFAULT_SOURCE_TYPE_POOL):
        raise ValueError(
            f"n_source_types must be <= {len(DEFAULT_SOURCE_TYPE_POOL)} "
            "(size of DEFAULT_SOURCE_TYPE_POOL)"
        )
    source_pool = DEFAULT_SOURCE_TYPE_POOL[:n_source_types]

    atoms: list[CorpusAtom] = []
    for topic in range(n_topics):
        for idx in range(n_atoms_per_topic):
            bid = _atom_id(topic, idx)
            session = f"sess_{rng.randrange(n_sessions):02d}"
            source_type = source_pool[rng.randrange(n_source_types)]
            atoms.append(
                CorpusAtom(
                    belief_id=bid,
                    topic=topic,
                    session_id=session,
                    source_type=source_type,
                )
            )

    # Edges: intra-topic dense, cross-topic sparse. Edge type cycled
    # across the TC-eligible set to give TC something interesting.
    edge_type_cycle = [EDGE_SUPPORTS, EDGE_CITES, EDGE_RELATES_TO]
    edges: list[Edge] = []
    by_topic: dict[int, list[str]] = {}
    for atom in atoms:
        by_topic.setdefault(atom.topic, []).append(atom.belief_id)

    for topic, ids in by_topic.items():
        for i, src in enumerate(ids):
            for j, dst in enumerate(ids):
                if i == j:
                    continue
                if rng.random() < intra_edge_density:
                    et = edge_type_cycle[(i + j) % len(edge_type_cycle)]
                    edges.append(Edge(src=src, dst=dst, type=et, weight=1.0))

    all_ids = [a.belief_id for a in atoms]
    for src in all_ids:
        for dst in all_ids:
            if src == dst:
                continue
            src_topic = src[:3]
            dst_topic = dst[:3]
            if src_topic == dst_topic:
                continue
            if rng.random() < cross_edge_density:
                edges.append(
                    Edge(src=src, dst=dst, type=EDGE_RELATES_TO, weight=0.5)
                )

    return SyntheticCorpus(atoms=tuple(atoms), edges=tuple(edges))


def populate_store(
    store: "MemoryStore",
    corpus: SyntheticCorpus,
    *,
    rng: random.Random,
    high_uncertainty_fraction: float = 0.15,
    timestamp: str = "2026-05-03T00:00:00Z",
) -> None:
    """Insert the corpus into a live ``MemoryStore``.

    α/β stamping: most atoms get Beta(1,1) (the v1.x default
    uninformative prior); ``high_uncertainty_fraction`` get a
    weaker Beta(0.7, 0.7) so their differential entropy clears
    the RW seed-floor reliably.
    """
    rng_local = random.Random(rng.random())
    for atom in corpus.atoms:
        if rng_local.random() < high_uncertainty_fraction:
            alpha, beta = 0.7, 0.7
        else:
            alpha, beta = 1.0, 1.0
        store.insert_belief(
            Belief(
                id=atom.belief_id,
                content=f"synthetic atom for topic {atom.topic}",
                content_hash=f"h_{atom.belief_id}",
                alpha=alpha,
                beta=beta,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE,
                locked_at=None,
                demotion_pressure=0,
                created_at=timestamp,
                last_retrieved_at=None,
                session_id=atom.session_id,
            )
        )
    for edge in corpus.edges:
        store.insert_edge(edge)


def feedback_verdict(
    composition: tuple[str, ...],
    corpus: SyntheticCorpus,
) -> str:
    """Return ``"confirm"`` if every belief in the composition
    shares a single topic; otherwise ``"junk"``.

    Single-topic agreement is the corpus's ground-truth predicate
    for "real relationship" — it's the simplest predicate the
    three candidate strategies all have a path to satisfy:

    * TC's shared-target shape pulls at most from the same target's
      incoming neighborhood, which is largely intra-topic at the
      default densities.
    * RW depth-2 walks tend to stay intra-topic because intra-edges
      dominate.
    * STS draws cross-session, so it must accidentally pick atoms
      from the same topic to confirm — which is the predicted
      weakness in the spec's H3.
    """
    if not composition:
        return "junk"
    first_topic = corpus.topic_of(composition[0])
    if first_topic is None:
        return "junk"
    for bid in composition[1:]:
        if corpus.topic_of(bid) != first_topic:
            return "junk"
    return "confirm"


def simulate_promotion(
    confirms: int,
    junks: int,
    *,
    initial_alpha: float = 1.0,
    initial_beta: float = 1.0,
) -> bool:
    """Return ``True`` if the phantom would promote under #196's gate.

    Each confirm increments α; each junk increments β. Promotion
    fires when ``α >= ALPHA_PROMOTION_THRESHOLD``.
    """
    alpha = initial_alpha + confirms
    beta = initial_beta + junks
    _ = beta  # documented for symmetry; gate is α-only per spec
    return alpha >= ALPHA_PROMOTION_THRESHOLD


__all__ = [
    "ALPHA_PROMOTION_THRESHOLD",
    "CorpusAtom",
    "SyntheticCorpus",
    "build_corpus",
    "feedback_verdict",
    "populate_store",
    "simulate_promotion",
]
