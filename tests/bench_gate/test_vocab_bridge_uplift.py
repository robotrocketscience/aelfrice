"""Bench gate for #433 HRR vocabulary bridge.

Measures the upstream invariant A2 depends on: on a labeled
``vocab_bridge`` corpus, a query that *should* recover canonical
entities (via the corpus's expected_canonicals annotation) gets at
least one of those canonicals appended by ``VocabBridge.rewrite``
above the noise floor.

A2's strict NDCG@k claim requires running the full retrieve_v2 path
against the labeled corpus and computing recall against ground-truth
relevant beliefs. That is the lab-side budget-rewrite-style follow-up;
this gate only checks the precondition (the bridge actually augments
queries it should augment) so a regression in harvest or rewrite logic
trips before the heavier NDCG measurement runs.

Skips on public CI (autouse `bench_gated` marker handles
``AELFRICE_CORPUS_ROOT`` absence). Skips again when the
``tests/corpus/v2_0/vocab_bridge/`` directory is empty.

Expected row schema (``tests/corpus/v2_0/vocab_bridge/*.jsonl``):

  {
    "id": "row-id",
    "query": "raw query string passed to retrieve",
    "store_beliefs": [
        {"id": "b1", "content": "...", "anchors": ["text", "..."]},
        ...
    ],
    "expected_canonicals": ["sqlite", "python", "..."]
  }

`store_beliefs[i].anchors` is optional; when present, each anchor
string is added as an inbound edge from a synthetic citing belief
to seed the bridge with anchor-source surface forms (#148 parity).
`expected_canonicals` is the set of canonical-entity tokens that
the bridge SHOULD append for this query; the test asserts at least
one is present in the rewritten output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    Belief,
    Edge,
)
from aelfrice.store import MemoryStore
from aelfrice.vocab_bridge import VocabBridge
from tests.conftest import load_corpus_module


def _mk_belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        demotion_pressure=0,
        created_at="2026-05-08T00:00:00Z",
        last_retrieved_at=None,
        retention_class="fact",
    )


def _seed_store(store_beliefs: list[dict[str, Any]]) -> MemoryStore:
    s = MemoryStore(":memory:")
    for entry in store_beliefs:
        bid = str(entry["id"])
        content = str(entry["content"])
        s.insert_belief(_mk_belief(bid, content))
    # Second pass for anchors so the citing-belief id space is stable
    # before edge insertion. Each anchor seeds one inbound edge from a
    # synthetic "anchor citer" so iter_incoming_anchor_text() yields
    # the surface form.
    for entry in store_beliefs:
        anchors = entry.get("anchors") or []
        for j, anchor in enumerate(anchors):
            citer_id = f"_a_{entry['id']}_{j}"
            if not s.get_belief(citer_id):
                s.insert_belief(_mk_belief(citer_id, "anchor citer"))
            s.insert_edge(Edge(
                src=citer_id,
                dst=str(entry["id"]),
                type="CITES",
                weight=1.0,
                anchor_text=str(anchor),
            ))
    return s


@pytest.mark.bench_gated
def test_bridge_appends_at_least_one_expected_canonical(
    aelfrice_corpus_root: Path,
) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "vocab_bridge")

    n_rows = 0
    n_hits = 0
    for row in rows:
        store = _seed_store(row["store_beliefs"])
        bridge = VocabBridge()
        bridge.build(store, store_path=f"/bench/{row['id']}")
        rewritten = bridge.rewrite(str(row["query"]))
        appended_tokens = set(rewritten.split()) - set(str(row["query"]).split())
        expected = {str(c).lower() for c in row.get("expected_canonicals", [])}
        if not expected:
            continue
        n_rows += 1
        if appended_tokens & expected:
            n_hits += 1

    if n_rows == 0:
        pytest.skip(
            "vocab_bridge corpus has no rows with expected_canonicals; "
            "annotate at least one row before this gate can fire"
        )

    # Loose threshold: a regression in harvest or rewrite that drops
    # ALL bridging is caught immediately. Tightening to a per-row
    # threshold (or NDCG@k uplift) is the A2-strict follow-up.
    coverage = n_hits / float(n_rows)
    assert coverage >= 0.5, (
        f"vocab_bridge appended ≥1 expected canonical on only "
        f"{n_hits}/{n_rows} rows ({coverage:.1%}); harvest/rewrite "
        f"regression suspected (was the surface-form pipeline broken?)"
    )
