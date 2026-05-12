"""Synthetic N=50k store builder for HRR cold-start bench gates (#697).

Produces a MemoryStore backed by an on-disk SQLite file with a deterministic
N=50k belief corpus and a sparse edge graph (~3 outgoing edges per belief).
The graph structure is intentionally non-empty so the HRR structural lane
has edges to index; an edgeless store would produce ``struct == 0`` and
make the bench meaningless.

Edge topology mirrors the perf gate in ``tests/test_hrr_struct_index.py``
AC6/AC7: three outgoing CITES edges per belief at offsets (1, 7, 31), giving
a sparse but globally connected graph that exercises the accumulate-and-norm
path in ``HRRStructIndex.build``.

Fixed seed: 42 (matches uri_baki_retest canonical default and the issue spec).
"""
from __future__ import annotations

import random
from pathlib import Path

from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CITES,
    EDGE_CONTRADICTS,
    LOCK_NONE,
    ORIGIN_UNKNOWN,
    RETENTION_UNKNOWN,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore


def build_n50k_store(
    memory_db: Path,
    *,
    n_beliefs: int = 50_000,
    seed: int = 42,
) -> MemoryStore:
    """Build a synthetic N=50k store for HRR cold-start bench tests.

    Creates (or opens) the SQLite database at ``memory_db``, inserts
    ``n_beliefs`` beliefs and a sparse edge graph, then returns the open
    ``MemoryStore``.  Call ``store.close()`` when done.

    The build is fully deterministic at fixed seed: same ``n_beliefs`` and
    ``seed`` always produce the same ``struct.npy`` byte-for-byte.

    Edge topology: three outgoing CITES edges per belief at offsets (1, 7, 31)
    plus a CONTRADICTS edge for every 100th belief (belief i → belief i+50),
    giving the structural lane real edge signal in both CITES and CONTRADICTS
    role vectors.
    """
    rng = random.Random(seed)

    store = MemoryStore(str(memory_db))
    for i in range(n_beliefs):
        bid = f"b{i:06d}"
        # Vary content slightly to avoid content_hash collisions.
        noise = rng.randint(0, 999_999)
        store.insert_belief(
            Belief(
                id=bid,
                content=f"belief {i} noise={noise}",
                content_hash=f"h_{bid}",
                alpha=1.0,
                beta=1.0,
                type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE,
                locked_at=None,
                demotion_pressure=0,
                created_at="2026-05-11T00:00:00Z",
                last_retrieved_at=None,
                session_id=None,
                origin=ORIGIN_UNKNOWN,
                corroboration_count=0,
                hibernation_score=None,
                activation_condition=None,
                retention_class=RETENTION_UNKNOWN,
            ),
        )

    # Sparse edge graph: 3 outgoing CITES per belief + 1 CONTRADICTS per 100.
    for i in range(n_beliefs):
        src = f"b{i:06d}"
        for off in (1, 7, 31):
            dst = f"b{(i + off) % n_beliefs:06d}"
            store.insert_edge(Edge(src=src, dst=dst, type=EDGE_CITES, weight=1.0))
        if i % 100 == 0:
            dst = f"b{(i + 50) % n_beliefs:06d}"
            store.insert_edge(
                Edge(src=src, dst=dst, type=EDGE_CONTRADICTS, weight=1.0)
            )

    return store
