"""#152 HRR structural-query lane — offline marker-routing recall ablation.

Builds a SYNTHETIC edge-connected corpus and reports `recall@k` of the
graph-connected answer with the structural lane OFF vs ON, driving
`retrieve_v2` directly (the same entry point the production `retrieve()`
shim delegates to after the #1107 Phase-5 graduation).

Each row is a `<KIND>:<target_id>` marker query whose answer — the belief
that points at `target_id` via an edge of `KIND` — shares NO vocabulary with
the marker string. With the lane OFF, `retrieve_v2` BM25-searches the literal
marker text and cannot reach the answer (recall ~0). With the lane ON, the
query parses as a structural marker and the HRR probe returns the edge source
(recall 1.0). This is the public, on-HEAD analogue of the #437
reproducibility-harness gate (whose 11/11 result flipped the resolver default
to True at v2.1) and the `tests/test_retrieve_v2_hrr_structural.py` IT1-IT6
wiring tests — added for the #1107 Phase-5 production graduation so the flip
carries a reproducible recall signal rather than resting only on prior gates.

The lane is marker-routed: on any non-marker query it falls through
byte-identically, so its production blast radius is confined to callers that
issue `<KIND>:<target_id>` queries (structured tooling), not free-text hook
prompts. No live-store content — deterministic and CI-safe.

Ship condition: `recall_uplift > 0`.

Run: `python benchmarks/hrr_structural_ablation.py`
"""
from __future__ import annotations

from dataclasses import dataclass

from aelfrice.hrr_index import HRRStructIndexCache
from aelfrice.models import (
    BELIEF_FACTUAL,
    EDGE_CONTRADICTS,
    EDGE_SUPPORTS,
    EDGE_CITES,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.retrieval import retrieve_v2
from aelfrice.store import MemoryStore

N_ROWS = 20
_NOISE_PER_ROW = 3
_BUDGET = 10_000
_KINDS = [EDGE_CONTRADICTS, EDGE_SUPPORTS, EDGE_CITES]

# Disjoint NATO-ish vocab so marker strings ("CONTRADICTS:tgt7") share no
# tokens with any belief content — the lane-off arm cannot BM25 its way in.
_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
]


def _content(seed: int) -> str:
    return " ".join(_WORDS[(seed + j) % len(_WORDS)] for j in range(5))


@dataclass
class _Result:
    n_rows: int
    recall_off: float
    recall_on: float

    @property
    def recall_uplift(self) -> float:
        return self.recall_on - self.recall_off


def _build(store: MemoryStore) -> list[tuple[str, str]]:
    """Populate the store; return the (marker_query, answer_id) list."""
    rows: list[tuple[str, str]] = []
    for i in range(N_ROWS):
        kind = _KINDS[i % len(_KINDS)]
        src, tgt = f"src{i}", f"tgt{i}"
        for bid, seed in ((src, i), (tgt, i + 100)):
            store.insert_belief(Belief(
                id=bid, content=_content(seed), content_hash=f"h_{bid}",
                alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE, locked_at=None,
                created_at="2026-06-01T00:00:00Z", last_retrieved_at=None,
            ))
        for j in range(_NOISE_PER_ROW):
            nid = f"noise{i}_{j}"
            store.insert_belief(Belief(
                id=nid, content=_content(i + 200 + j), content_hash=f"h_{nid}",
                alpha=1.0, beta=1.0, type=BELIEF_FACTUAL,
                lock_level=LOCK_NONE, locked_at=None,
                created_at="2026-06-01T00:00:00Z", last_retrieved_at=None,
            ))
        store.insert_edge(Edge(src=src, dst=tgt, type=kind, weight=1.0))
        rows.append((f"{kind}:{tgt}", src))
    return rows


def _recall(store: MemoryStore, rows: list[tuple[str, str]], *,
            use_lane: bool, k: int) -> float:
    cache = HRRStructIndexCache(store=store, dim=512, seed=42) if use_lane else None
    hits = 0
    for query, answer in rows:
        got = [
            b.id for b in retrieve_v2(
                store, query, use_hrr_structural=use_lane,
                hrr_struct_index_cache=cache, budget=_BUDGET,
            ).beliefs[:k]
        ]
        if answer in got:
            hits += 1
    return hits / len(rows)


def main() -> None:
    store = MemoryStore(":memory:")
    try:
        rows = _build(store)
        res = _Result(
            n_rows=len(rows),
            recall_off=_recall(store, rows, use_lane=False, k=3),
            recall_on=_recall(store, rows, use_lane=True, k=3),
        )
    finally:
        store.close()
    print(f"synthetic corpus: {res.n_rows} marker queries "
          f"({N_ROWS} edges, {_NOISE_PER_ROW} noise/row), budget={_BUDGET}, k=3")
    print(f"  recall@k  OFF={res.recall_off:.3f} ON={res.recall_on:.3f} "
          f"uplift={res.recall_uplift:+.3f}")
    ship = res.recall_uplift > 0
    print(f"  ship condition (recall uplift > 0): {'PASS' if ship else 'FAIL'}")


if __name__ == "__main__":
    main()
