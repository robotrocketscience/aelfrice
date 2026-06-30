"""Core scoring for the pollution-recovery benchmark (#1011).

Each fixture case is: a query, one or more user-stated `facts` that
answer it, and a set of keyword-overlapping document `chunks` that
should NOT out-rank the facts. We build a fresh in-memory store per
case, insert facts as `user_stated` and chunks as `document_recent`,
run `retrieve()`, and score how well the user facts survive ranking.

Metrics (higher recall / lower rank / lower chunk-share = better):
  * fact_recall_at_k : fraction of cases whose answering fact is in top-k
  * mean_fact_rank   : mean 1-based rank of the answering fact (inf→cap)
  * chunk_share_at_k : mean fraction of top-k slots held by doc chunks
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_DOCUMENT_RECENT,
    ORIGIN_USER_STATED,
    Belief,
)
from aelfrice.retrieval import retrieve
from aelfrice.store import MemoryStore

# Rank assigned when the answering fact does not appear at all — a finite
# cap so mean_fact_rank stays defined and monotone.
_MISSING_RANK: int = 999


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def _mk(bid: str, content: str, origin: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=_hash(content),
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-06-01T00:00:00+00:00",
        last_retrieved_at=None,
        origin=origin,
    )


@dataclass(frozen=True)
class CaseResult:
    query: str
    regime: str               # lexical | entity | weak (fixture label)
    fact_rank: int            # 1-based rank of the best answering fact
    fact_in_top_k: bool
    chunk_share_top_k: float  # fraction of top-k that are doc chunks


@dataclass(frozen=True)
class Report:
    n_cases: int
    k: int
    use_origin_tier_rerank: bool
    fact_recall_at_k: float
    mean_fact_rank: float
    chunk_share_at_k: float
    cases: tuple[CaseResult, ...]


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _score_case(
    case: dict[str, Any], *, k: int, use_origin_tier_rerank: bool
) -> CaseResult:
    fact_ids = {f["id"] for f in case["facts"]}
    relevant = set(case["relevant_fact_ids"])
    store = MemoryStore(":memory:")
    try:
        for f in case["facts"]:
            store.insert_belief(_mk(f["id"], f["content"], ORIGIN_USER_STATED))
        for c in case["chunks"]:
            store.insert_belief(
                _mk(c["id"], c["content"], ORIGIN_DOCUMENT_RECENT)
            )
        # Large budget so ranking — not the budget trim — decides order.
        kwargs: dict[str, Any] = {"token_budget": 100_000}
        if use_origin_tier_rerank:
            kwargs["use_origin_tier_rerank"] = True
        hits = retrieve(store, case["query"], **kwargs)
    finally:
        store.close()
    order = [h.id for h in hits]
    # best (lowest) rank among the relevant answering facts
    ranks = [order.index(fid) + 1 for fid in relevant if fid in order]
    fact_rank = min(ranks) if ranks else _MISSING_RANK
    top_k = order[:k]
    chunk_hits = sum(1 for bid in top_k if bid not in fact_ids)
    chunk_share = chunk_hits / k if k else 0.0
    return CaseResult(
        query=case["query"],
        regime=str(case.get("regime", "unknown")),
        fact_rank=fact_rank,
        fact_in_top_k=fact_rank <= k,
        chunk_share_top_k=chunk_share,
    )


def regime_recall(report: "Report") -> dict[str, float]:
    """Per-regime fact_recall@k — the characterization headline.

    Documents which retrieval regimes survive doc-chunk pollution
    (lexical/entity) and which do not (weak), per the #1011 R&D.
    """
    by: dict[str, list[bool]] = {}
    for r in report.cases:
        by.setdefault(r.regime, []).append(r.fact_in_top_k)
    return {
        regime: (sum(hits) / len(hits) if hits else 0.0)
        for regime, hits in sorted(by.items())
    }


def evaluate(
    cases: list[dict[str, Any]],
    *,
    k: int = 5,
    use_origin_tier_rerank: bool = False,
) -> Report:
    results = [
        _score_case(c, k=k, use_origin_tier_rerank=use_origin_tier_rerank)
        for c in cases
    ]
    n = len(results) or 1
    return Report(
        n_cases=len(results),
        k=k,
        use_origin_tier_rerank=use_origin_tier_rerank,
        fact_recall_at_k=sum(r.fact_in_top_k for r in results) / n,
        mean_fact_rank=sum(r.fact_rank for r in results) / n,
        chunk_share_at_k=sum(r.chunk_share_top_k for r in results) / n,
        cases=tuple(results),
    )
