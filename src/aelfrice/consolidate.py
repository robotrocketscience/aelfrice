"""Audit-only near-duplicate cluster report (#1312, #1176 proposal 4).

This module is the **algorithm**. The audit-only CLI surface
(`aelf doctor --consolidate`) lives in `cli.py`. There is no write
path here and none is planned under this issue: the operator funded the
report and explicitly did not fund contraction. Nothing in this module
inserts an edge, soft-deletes a belief, or writes a log row.

Same division of labour `dedup.py` already uses, and the same
similarity predicate — a pair is a duplicate iff Jaccard >=
`jaccard_min` **and** Levenshtein ratio >= `levenshtein_min`. Reusing
the shipped predicate is deliberate: this report exists to say what
contraction *would* do at the thresholds the product already ships, so
inventing a second predicate would make the answer unfalsifiable.

Two things differ from `dedup.py`, both because this report is about
the whole store rather than a rendered sample:

  * **Candidate pairs come from 4-gram blocking, not O(n^2).** `dedup`
    budgets direct Jaccard prefiltering at a ~1.6k-belief median; at the
    44.5k-belief scale this report targets that is ~991M pairs, which is
    not viable. Blocking pairs beliefs that share at least one order-4
    token shingle whose document frequency is `<= max_shingle_df`, which
    is standard MinHash-free LSH blocking. Skipped high-df shingles are
    **counted and reported** (`n_shingles_over_df`), never dropped
    silently — a blocking cap that does not report itself reads as full
    coverage when it is not.

  * **Only components of size >= 3 are reported.** Two-member components
    are plain supersession and are not what consolidation is for.

The parent of a component is its **medoid** — the member minimising the
summed Levenshtein distance to the other members, tie-broken on belief
id ASC. The medoid is an existing belief chosen by a closed-form
criterion: no generation, no model, no LLM, and therefore replayable.
`dedup.DuplicateCluster` names its representative `min(member_ids)`
instead, which is deterministic but arbitrary; the medoid is the member
a contraction would actually want to keep.

Determinism: the report is a pure function of the active belief set.
Shingles are built from `bm25.tokenize` (the production vocabulary),
components come from an order-independent union-find, medoids have an
explicit id-ASC tiebreak, and every emitted list is sorted. No clock, no
env, no randomness, no Python `hash`.

Aggregate counts only — callers render no belief content, no ids and no
paths, so the report is safe to paste into an issue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from aelfrice.bm25 import tokenize
from aelfrice.dedup import (
    DEFAULT_JACCARD_MIN,
    DEFAULT_LEVENSHTEIN_MIN,
    jaccard,
    levenshtein_distance,
    levenshtein_ratio,
)
from aelfrice.store import MemoryStore

SHINGLE_N: Final[int] = 4
"""Token shingle width used for candidate blocking."""

DEFAULT_MAX_SHINGLE_DF: Final[int] = 32
"""Skip 4-grams appearing in more than this many beliefs when blocking.

A 4-gram shared by hundreds of beliefs contributes a quadratic number of
candidate pairs and almost no signal — two beliefs that are genuinely
near-duplicate share many *rarer* shingles as well, so they survive the
cap through those. Measured on the development store, tightening this
from 400 to 32 left the resulting components bit-identical.
"""

MIN_COMPONENT_SIZE: Final[int] = 3
"""Smallest component the report counts. Pairs are plain supersession."""


@dataclass(frozen=True)
class ConsolidationCluster:
    """One connected component of near-duplicate beliefs.

    `member_ids` is sorted lexicographically. `medoid_id` is the member
    minimising summed Levenshtein distance to the other members, with an
    id-ASC tiebreak — the belief a contraction would keep.
    """

    medoid_id: str
    member_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.member_ids)


@dataclass(frozen=True)
class ConsolidationReport:
    """Summary of one audit pass.

    `n_would_remove` is the count of beliefs a contraction would retire:
    one medoid survives per component, so it is
    `n_beliefs_in_clusters - n_clusters`. It is the number that prices
    the intervention, which is why it is computed here rather than left
    to the caller to derive.
    """

    n_beliefs_scanned: int
    n_shingles_over_df: int
    n_candidate_pairs: int
    n_duplicate_pairs: int
    n_clusters: int
    n_beliefs_in_clusters: int
    largest_cluster: int
    jaccard_min: float
    levenshtein_min: float
    max_shingle_df: int
    clusters: tuple[ConsolidationCluster, ...] = field(default=())

    @property
    def n_would_remove(self) -> int:
        return self.n_beliefs_in_clusters - self.n_clusters

    @property
    def share_of_store(self) -> float:
        """`n_would_remove` as a percentage of the beliefs scanned."""
        if self.n_beliefs_scanned == 0:
            return 0.0
        return 100.0 * self.n_would_remove / self.n_beliefs_scanned


def shingles(tokens: list[str]) -> frozenset[tuple[str, ...]]:
    """Order-4 token shingles. Beliefs shorter than 4 tokens yield one
    shingle holding the whole token tuple, so they can still block."""
    if not tokens:
        return frozenset()
    if len(tokens) < SHINGLE_N:
        return frozenset({tuple(tokens)})
    return frozenset(
        tuple(tokens[i : i + SHINGLE_N])
        for i in range(len(tokens) - SHINGLE_N + 1)
    )


def _medoid(member_ids: list[str], content: dict[str, str]) -> str:
    """The member minimising summed Levenshtein distance to the others.

    Ties break on belief id ASC, so the choice is total. `member_ids`
    is assumed sorted; iterating it in order makes the strict `<`
    comparison below resolve ties toward the smaller id without a
    second key.
    """
    best_id = member_ids[0]
    best_cost: int | None = None
    for candidate in member_ids:
        cost = sum(
            levenshtein_distance(content[candidate], content[other])
            for other in member_ids
            if other != candidate
        )
        if best_cost is None or cost < best_cost:
            best_cost, best_id = cost, candidate
    return best_id


def _components(
    pairs: list[tuple[str, str]],
) -> list[list[str]]:
    """Union-find collapse of duplicate pairs into components.

    Order-independent: parents always point at the lexicographically
    smaller root, so the component set does not depend on pair order.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            lo, hi = (ra, rb) if ra < rb else (rb, ra)
            parent[hi] = lo

    groups: dict[str, set[str]] = {}
    for a, b in pairs:
        for member in (a, b):
            groups.setdefault(find(member), set()).add(member)
    return [sorted(g) for g in groups.values()]


def consolidation_audit(
    store: MemoryStore,
    *,
    jaccard_min: float = DEFAULT_JACCARD_MIN,
    levenshtein_min: float = DEFAULT_LEVENSHTEIN_MIN,
    max_shingle_df: int = DEFAULT_MAX_SHINGLE_DF,
) -> ConsolidationReport:
    """Cluster active beliefs and report what a contraction would remove.

    Read-only. Raises `ValueError` on malformed thresholds so a typo
    fails loudly rather than silently scanning at the wrong gate.
    """
    if not 0.0 <= jaccard_min <= 1.0:
        raise ValueError(
            f"jaccard_min must be in [0.0, 1.0], got {jaccard_min}"
        )
    if not 0.0 <= levenshtein_min <= 1.0:
        raise ValueError(
            f"levenshtein_min must be in [0.0, 1.0], got {levenshtein_min}"
        )
    if max_shingle_df < 2:
        raise ValueError(f"max_shingle_df must be >= 2, got {max_shingle_df}")

    rows = store.list_beliefs_for_indexing()
    content = {bid: (text or "") for bid, text in rows}

    empty = ConsolidationReport(
        n_beliefs_scanned=len(rows),
        n_shingles_over_df=0,
        n_candidate_pairs=0,
        n_duplicate_pairs=0,
        n_clusters=0,
        n_beliefs_in_clusters=0,
        largest_cluster=0,
        jaccard_min=jaccard_min,
        levenshtein_min=levenshtein_min,
        max_shingle_df=max_shingle_df,
    )
    # Two beliefs are enough to form a *pair*, which the report counts
    # even though it takes three to form a cluster. Guarding on
    # MIN_COMPONENT_SIZE here would report `n_duplicate_pairs=0` on a
    # two-belief store that plainly has one.
    if len(rows) < 2:
        return empty

    tokens = {bid: tokenize(text) for bid, text in content.items()}
    token_sets = {bid: frozenset(toks) for bid, toks in tokens.items()}

    postings: dict[tuple[str, ...], list[str]] = {}
    for bid, toks in tokens.items():
        for shingle in shingles(toks):
            postings.setdefault(shingle, []).append(bid)

    candidates: set[tuple[str, str]] = set()
    n_over_df = 0
    for members in postings.values():
        if len(members) < 2:
            continue
        if len(members) > max_shingle_df:
            n_over_df += 1
            continue
        members.sort()
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidates.add((members[i], members[j]))

    duplicates = [
        (a, b)
        for a, b in sorted(candidates)
        if jaccard(token_sets[a], token_sets[b]) >= jaccard_min
        and levenshtein_ratio(content[a], content[b]) >= levenshtein_min
    ]

    groups = [
        g for g in _components(duplicates) if len(g) >= MIN_COMPONENT_SIZE
    ]
    groups.sort(key=lambda g: (-len(g), g[0]))

    clusters = tuple(
        ConsolidationCluster(
            medoid_id=_medoid(g, content), member_ids=tuple(g)
        )
        for g in groups
    )
    in_clusters = sum(c.size for c in clusters)

    return ConsolidationReport(
        n_beliefs_scanned=len(rows),
        n_shingles_over_df=n_over_df,
        n_candidate_pairs=len(candidates),
        n_duplicate_pairs=len(duplicates),
        n_clusters=len(clusters),
        n_beliefs_in_clusters=in_clusters,
        largest_cluster=max((c.size for c in clusters), default=0),
        jaccard_min=jaccard_min,
        levenshtein_min=levenshtein_min,
        max_shingle_df=max_shingle_df,
        clusters=clusters,
    )


def format_consolidation_report(report: ConsolidationReport) -> str:
    """Render the report. Aggregate counts only — no belief content, no
    ids, no paths, so the output is safe to paste into an issue."""
    lines = [
        "Consolidation audit (read-only — nothing was written)",
        f"  thresholds            : Jaccard >= {report.jaccard_min}, "
        f"Levenshtein ratio >= {report.levenshtein_min}",
        f"  active beliefs        : {report.n_beliefs_scanned:,}",
        f"  4-grams over df={report.max_shingle_df:<4d}  : "
        f"{report.n_shingles_over_df:,} skipped by blocking",
        f"  candidate pairs       : {report.n_candidate_pairs:,}",
        f"  duplicate pairs       : {report.n_duplicate_pairs:,}",
        f"  clusters (size >= {MIN_COMPONENT_SIZE})  : {report.n_clusters:,}",
        f"  beliefs in a cluster  : {report.n_beliefs_in_clusters:,}",
        f"  largest cluster       : {report.largest_cluster:,}",
        f"  would remove          : {report.n_would_remove:,} "
        f"({report.share_of_store:.2f}% of active)",
    ]
    if report.n_clusters:
        sizes = sorted((c.size for c in report.clusters), reverse=True)[:10]
        lines.append(
            "  ten largest sizes     : " + ", ".join(str(s) for s in sizes)
        )
    lines.append(
        "  NOTE: audit only. Contraction is not funded (#1312); no "
        "SUPERSEDES edge is written and no belief is retired."
    )
    return "\n".join(lines)
