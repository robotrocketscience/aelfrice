"""uri_baki retest harness — issue #153.

The prior synthetic-graph attempt regressed NDCG@10 by ~0.05 because
the locked set was random-sampled (uncorrelated with query relevance)
and supersession / recency had no correlation with relevance either.
This harness fixes the methodology by giving each effect the kind of
signal it would carry in a real production corpus:

- **Locked floor.** A locked belief is, by construction, relevant to
  the user's query distribution: the user pinned it because they
  care about that topic. Modelled here as: locks fall only on
  beliefs whose topic is in the **hot-topic** subset, which is also
  where most queries land.
- **Supersession demote.** ``SUPERSEDED_BY`` in production typically
  links an old version of a claim to a newer, better version of the
  same claim. Modelled here as: superseded beliefs and their
  successors share a topic. A query on that topic should rank the
  successor above the superseded.
- **Recency decay.** Older beliefs are more likely to be stale.
  Modelled here by making cold-topic beliefs skew older while
  hot-topic beliefs skew newer. A naïve term-overlap ranker can
  surface an old, off-topic belief that incidentally matches a query
  term; recency decay pushes it down.

The harness still prints a ``random_locks`` row for the locks effect
to reproduce the prior negative as a control. The ratification gate
in the issue body is **ΔNDCG@10 ≥ +0.02 absolute, latency ≤ 5 ms**
on the relevance-aware row at N=10k.

This module is offline-only and not wired into ``aelf bench``.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from aelfrice.models import (
    LOCK_NONE,
    LOCK_USER,
    ORIGIN_UNKNOWN,
    RETENTION_UNKNOWN,
    Belief,
)
from aelfrice.uri_baki import (
    apply_locked_floor,
    apply_recency_decay,
    apply_supersession_demote,
)


# ---------------------------------------------------------------------------
# Synthetic corpus
# ---------------------------------------------------------------------------


@dataclass
class CorpusConfig:
    n_beliefs: int = 10_000
    n_queries: int = 200
    n_topics: int = 50
    n_hot_topics: int = 10
    hot_query_fraction: float = 0.8  # 80% of queries land on hot topics
    topic_vocab_size: int = 30
    filler_vocab_size: int = 500
    k_topical: int = 3
    k_filler: int = 8
    k_query_topical: int = 2
    k_query_filler: int = 2
    p_locked: float = 0.005
    p_superseded: float = 0.05
    # Noise: fraction of beliefs whose "topical" terms come from a
    # *different* topic. Models off-topic beliefs that incidentally
    # share vocabulary with a query (the failure mode in #281).
    p_off_topic_noise: float = 0.15
    mean_age_days_hot: float = 30.0
    mean_age_days_cold: float = 365.0
    seed: int = 42


@dataclass
class SyntheticCorpus:
    beliefs: list[Belief]
    belief_topics: list[int]
    belief_terms: list[set[str]]
    queries: list[set[str]]
    query_topics: list[int]
    superseded_ids: set[str]
    df: dict[str, int]
    locked_ids_relevance_aware: set[str]
    locked_ids_random: set[str]
    ref_now: datetime = field(
        default_factory=lambda: datetime(2026, 5, 1, tzinfo=timezone.utc),
    )


def build_corpus(cfg: CorpusConfig) -> SyntheticCorpus:
    rng = random.Random(cfg.seed)
    ref_now = datetime(2026, 5, 1, tzinfo=timezone.utc)

    hot_topics = list(range(cfg.n_hot_topics))
    cold_topics = list(range(cfg.n_hot_topics, cfg.n_topics))

    topic_vocabs: list[list[str]] = [
        [f"t{ti}_w{wi}" for wi in range(cfg.topic_vocab_size)]
        for ti in range(cfg.n_topics)
    ]
    filler_vocab = [f"f_{i}" for i in range(cfg.filler_vocab_size)]

    # Queries concentrated on hot topics.
    query_topics: list[int] = []
    for _ in range(cfg.n_queries):
        if rng.random() < cfg.hot_query_fraction:
            query_topics.append(rng.choice(hot_topics))
        else:
            query_topics.append(rng.choice(cold_topics))
    queries: list[set[str]] = []
    for qt in query_topics:
        topical = rng.sample(topic_vocabs[qt], cfg.k_query_topical)
        filler = rng.sample(filler_vocab, cfg.k_query_filler)
        queries.append(set(topical) | set(filler))

    # Beliefs.
    beliefs: list[Belief] = []
    belief_topics: list[int] = []
    belief_terms: list[set[str]] = []
    df: dict[str, int] = {}
    hot_indices: list[int] = []
    topic_to_indices: dict[int, list[int]] = {ti: [] for ti in range(cfg.n_topics)}

    for i in range(cfg.n_beliefs):
        # Beliefs: 50% on hot topics, 50% on cold (so hot topics are
        # densely covered relative to cold).
        if rng.random() < 0.5:
            topic = rng.choice(hot_topics)
            mean_age = cfg.mean_age_days_hot
        else:
            topic = rng.choice(cold_topics)
            mean_age = cfg.mean_age_days_cold
        # Off-topic noise: with probability p_off_topic_noise, draw
        # topical terms from a *different* topic. Belief is still
        # "owned" by `topic` for ground-truth purposes, but its
        # vocabulary makes the ranker mis-attribute it.
        vocab_topic = topic
        if rng.random() < cfg.p_off_topic_noise:
            others = [t for t in range(cfg.n_topics) if t != topic]
            vocab_topic = rng.choice(others)
        topical = rng.sample(topic_vocabs[vocab_topic], cfg.k_topical)
        filler = rng.sample(filler_vocab, cfg.k_filler)
        terms = set(topical) | set(filler)
        belief_terms.append(terms)
        belief_topics.append(topic)
        for t in terms:
            df[t] = df.get(t, 0) + 1
        age_days = rng.expovariate(1.0 / mean_age)
        created_at = (ref_now - timedelta(days=age_days)).isoformat()
        bid = f"b{i:06d}"
        beliefs.append(
            Belief(
                id=bid,
                content=" ".join(sorted(terms)),
                content_hash=bid,
                alpha=1.0,
                beta=1.0,
                type="factual",
                lock_level=LOCK_NONE,
                locked_at=None,
                created_at=created_at,
                last_retrieved_at=None,
                session_id=None,
                origin=ORIGIN_UNKNOWN,
                corroboration_count=0,
                hibernation_score=None,
                activation_condition=None,
                retention_class=RETENTION_UNKNOWN,
            ),
        )
        if topic in hot_topics:
            hot_indices.append(i)
        topic_to_indices[topic].append(i)

    # Relevance-aware locks: only on hot-topic beliefs.
    n_to_lock = int(round(cfg.p_locked * cfg.n_beliefs))
    n_to_lock = min(n_to_lock, len(hot_indices))
    relevance_aware_lock_idx = (
        rng.sample(hot_indices, n_to_lock) if n_to_lock else []
    )
    locked_ids_relevance_aware = {
        beliefs[i].id for i in relevance_aware_lock_idx
    }
    # Control: random locks (ignoring topic). Same count.
    random_lock_idx = (
        rng.sample(range(cfg.n_beliefs), n_to_lock) if n_to_lock else []
    )
    locked_ids_random = {beliefs[i].id for i in random_lock_idx}

    # Supersession with same-topic successors. Pick a random pair
    # (a, b) within the same topic; mark `a` as superseded by `b`.
    # Both remain in the corpus (the ranker doesn't know about
    # supersession unless the post-rank effect tells it).
    superseded_idx: set[int] = set()
    n_sup_target = int(round(cfg.p_superseded * cfg.n_beliefs))
    attempts = 0
    while len(superseded_idx) < n_sup_target and attempts < n_sup_target * 5:
        attempts += 1
        topic = rng.randrange(cfg.n_topics)
        members = topic_to_indices[topic]
        if len(members) < 2:
            continue
        a, b = rng.sample(members, 2)
        if a in superseded_idx:
            continue
        superseded_idx.add(a)
        # b is the (implicit) successor; we don't materialise edges,
        # we just keep both in the corpus. The post-rank demote uses
        # the superseded set only.
    superseded_ids = {beliefs[i].id for i in superseded_idx}

    return SyntheticCorpus(
        beliefs=beliefs,
        belief_topics=belief_topics,
        belief_terms=belief_terms,
        queries=queries,
        query_topics=query_topics,
        superseded_ids=superseded_ids,
        df=df,
        locked_ids_relevance_aware=locked_ids_relevance_aware,
        locked_ids_random=locked_ids_random,
        ref_now=ref_now,
    )


# ---------------------------------------------------------------------------
# Ranker + ground truth
# ---------------------------------------------------------------------------


def _idf(df: dict[str, int], term: str, n: int) -> float:
    return math.log((n + 1.0) / (df.get(term, 0) + 1.0)) + 1.0


def baseline_scores(
    query: set[str],
    belief_terms: Sequence[set[str]],
    df: dict[str, int],
    n: int,
) -> list[float]:
    idf_map = {t: _idf(df, t, n) for t in query}
    out: list[float] = []
    for terms in belief_terms:
        score = 0.0
        for t in query:
            if t in terms:
                score += idf_map[t]
        out.append(score)
    return out


def ndcg_at_k(
    scores: Sequence[float], relevance: Sequence[int], k: int = 10,
) -> float:
    ranked = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True,
    )[:k]
    dcg = 0.0
    for rank, idx in enumerate(ranked, start=1):
        gain = relevance[idx]
        if gain:
            dcg += gain / math.log2(rank + 1)
    n_rel = sum(relevance)
    ideal_k = min(n_rel, k)
    if ideal_k == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_k + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


@dataclass
class ConditionResult:
    name: str
    ndcg_at_10: float
    delta_vs_baseline: float
    ms_per_query: float


def _set_locks(
    beliefs: list[Belief], lock_ids: set[str],
) -> list[Belief]:
    """Return a new belief list with locks applied per ``lock_ids``.

    Avoids mutating the corpus so we can run multiple lock scenarios
    against the same beliefs.
    """
    out: list[Belief] = []
    for b in beliefs:
        out.append(
            Belief(
                id=b.id, content=b.content, content_hash=b.content_hash,
                alpha=b.alpha, beta=b.beta, type=b.type,
                lock_level=LOCK_USER if b.id in lock_ids else LOCK_NONE,
                locked_at=b.locked_at,
                created_at=b.created_at,
                last_retrieved_at=b.last_retrieved_at,
                session_id=b.session_id, origin=b.origin,
                corroboration_count=b.corroboration_count,
                hibernation_score=b.hibernation_score,
                activation_condition=b.activation_condition,
                retention_class=b.retention_class,
            ),
        )
    return out


def run_condition(
    name: str,
    corpus: SyntheticCorpus,
    *,
    lock_ids: set[str] | None = None,
    apply_floor: bool = False,
    apply_demote: bool = False,
    apply_decay: bool = False,
    floor_quantile: float = 0.5,
    demote_factor: float = 0.5,
    decay_lambda: float = 1.0 / 180.0,
) -> ConditionResult:
    n = len(corpus.beliefs)
    locked_beliefs = _set_locks(
        corpus.beliefs, lock_ids if lock_ids is not None else set(),
    )
    ndcg_total = 0.0
    elapsed_ns = 0
    for q, qt in zip(corpus.queries, corpus.query_topics, strict=True):
        relevance = [1 if t == qt else 0 for t in corpus.belief_topics]
        t0 = time.perf_counter_ns()
        scores = baseline_scores(q, corpus.belief_terms, corpus.df, n)
        if apply_demote:
            scores = apply_supersession_demote(
                locked_beliefs, scores, corpus.superseded_ids,
                factor=demote_factor,
            )
        if apply_decay:
            scores = apply_recency_decay(
                locked_beliefs, scores,
                now=corpus.ref_now, lam=decay_lambda,
            )
        if apply_floor:
            matched = sorted(s for s in scores if s > 0.0)
            if matched:
                idx = int(floor_quantile * (len(matched) - 1))
                floor = matched[idx]
            else:
                floor = 0.0
            scores = apply_locked_floor(
                locked_beliefs, scores, floor=floor,
            )
        elapsed_ns += time.perf_counter_ns() - t0
        ndcg_total += ndcg_at_k(scores, relevance, k=10)
    n_q = len(corpus.queries)
    return ConditionResult(
        name=name,
        ndcg_at_10=ndcg_total / n_q,
        delta_vs_baseline=0.0,  # filled in by caller
        ms_per_query=(elapsed_ns / n_q) / 1e6,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(cfg: CorpusConfig) -> dict[str, object]:
    corpus = build_corpus(cfg)
    base = run_condition("baseline", corpus)
    conditions: list[ConditionResult] = []

    # Locked floor: relevance-aware + a control with random locks.
    for floor_q in (0.5, 0.75, 0.9):
        r = run_condition(
            f"locked_floor[ra,q={floor_q}]", corpus,
            lock_ids=corpus.locked_ids_relevance_aware,
            apply_floor=True, floor_quantile=floor_q,
        )
        conditions.append(r)
    r = run_condition(
        "locked_floor[random,q=0.5]", corpus,
        lock_ids=corpus.locked_ids_random,
        apply_floor=True, floor_quantile=0.5,
    )
    conditions.append(r)

    # Supersession demote sweep.
    for factor in (0.7, 0.5, 0.25):
        r = run_condition(
            f"supersession_demote[f={factor}]", corpus,
            apply_demote=True, demote_factor=factor,
        )
        conditions.append(r)

    # Recency decay sweep.
    for lam in (1 / 365, 1 / 180, 1 / 90):
        r = run_condition(
            f"recency_decay[λ={lam:.5f}]", corpus,
            apply_decay=True, decay_lambda=lam,
        )
        conditions.append(r)

    # Combined (relevance-aware locks, default factor and lambda).
    r = run_condition(
        "combined[ra,q=0.5,f=0.5,λ=1/180]", corpus,
        lock_ids=corpus.locked_ids_relevance_aware,
        apply_floor=True, floor_quantile=0.5,
        apply_demote=True, demote_factor=0.5,
        apply_decay=True, decay_lambda=1 / 180,
    )
    conditions.append(r)

    for r in conditions:
        r.delta_vs_baseline = r.ndcg_at_10 - base.ndcg_at_10

    return {
        "config": cfg.__dict__,
        "n_beliefs": cfg.n_beliefs,
        "n_queries": cfg.n_queries,
        "n_locked_relevance_aware": len(corpus.locked_ids_relevance_aware),
        "n_superseded": len(corpus.superseded_ids),
        "baseline": base.__dict__,
        "conditions": [r.__dict__ for r in conditions],
    }


def format_table(report: dict[str, object]) -> str:
    base = report["baseline"]
    lines = [
        f"N={report['n_beliefs']}  Q={report['n_queries']}  "
        f"locks(ra)={report['n_locked_relevance_aware']}  "
        f"superseded={report['n_superseded']}",
        "",
        f"{'condition':<36} {'NDCG@10':>9} {'Δ vs base':>10} "
        f"{'ms/query':>10}",
        "-" * 70,
        f"{'baseline':<36} {base['ndcg_at_10']:>9.4f} {'—':>10} "
        f"{base['ms_per_query']:>10.3f}",
    ]
    for cond in report["conditions"]:
        lines.append(
            f"{cond['name']:<36} {cond['ndcg_at_10']:>9.4f} "
            f"{cond['delta_vs_baseline']:>+10.4f} "
            f"{cond['ms_per_query']:>10.3f}",
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-beliefs", type=int, default=10_000)
    p.add_argument("--n-queries", type=int, default=200)
    p.add_argument("--n-topics", type=int, default=50)
    p.add_argument("--n-hot-topics", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--json", default="")
    args = p.parse_args()
    cfg = CorpusConfig(
        n_beliefs=args.n_beliefs,
        n_queries=args.n_queries,
        n_topics=args.n_topics,
        n_hot_topics=args.n_hot_topics,
        seed=args.seed,
    )
    report = run(cfg)
    print(format_table(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
