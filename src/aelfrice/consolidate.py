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
    not viable. A belief is posted to every shingle whose document
    frequency is `<= max_shingle_df`, which is standard MinHash-free LSH
    blocking, and candidate pairs are drawn from the resulting buckets.

    The cap has one cliff, which is #1316. It was justified by the claim
    that genuine near-duplicates "share many rarer shingles as well, so
    they survive the cap through those." That is false exactly where it
    matters most: in a *homogeneous* family every member shares every
    shingle, so all of them sit at `df = K` and none is rarer. Past the
    cap the whole family lost all its postings and the report announced
    **zero** duplicates precisely when the duplicate family was largest
    — at the shipped cap of 32, a 32-member family reported 31 removable
    and a 33-member family reported 0.

    So beliefs the cap leaves with **no posting at all** are rescued
    onto their rarest shared shingles, and a homogeneous family becomes
    its own bucket at any size. The rescue is a fallback rather than a
    replacement, which matters: applying rarest-shingle posting to every
    belief fixes the homogeneous cliff and opens a *heterogeneous* one,
    because two near-duplicates whose minimum `df` differs then never
    share a bucket at all. Measured, that variant dropped 490 beliefs
    and 81 whole clusters the cap had found. As a fallback the candidate
    set is a strict superset of the cap's, so no cluster can be lost —
    see `_blocked_pairs`. `n_beliefs_rescued` reports how many beliefs
    took the fallback, so the mechanism is never silent.

    The residual cost risk is a corpus of near-identical boilerplate,
    where one bucket is quadratic in its own size. That is bounded by
    `max_candidate_pairs` — counted as pairs *attempted*, so the bound
    holds even when the buckets keep re-proposing pairs already held —
    rather than by discarding evidence, and hitting the bound sets
    `truncated`, because a budget that does not report itself reads as
    full coverage when it is not. Buckets are consumed smallest-first,
    so truncation drops the largest, least discriminating ones.

    **Known residual, measured rather than assumed.** The rescue fires
    only for beliefs the cap leaves with *no* posting. A member of an
    over-cap family that happens to carry one low-`df` shingle — a
    version string two of them share, say — posts there instead, never
    reaches the family's own bucket, and can still be split off. The
    fallback does not close that; only removing the cap entirely would.
    Priced on the development store (`benchmarks/consolidate_blocking_recall.py`,
    2026-08-04): blocking with no cap at all clusters 1,669 beliefs
    against the shipped 1,638, so the whole remaining gap is **31
    beliefs of 44,594** — 98.1% of the reachable set, against a cap that
    is the only thing standing between a boilerplate-heavy store and a
    quadratic pass. Left as a tradeoff, not a defect.

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

`n_would_remove` excludes user-locked members. Contraction keeps one
medoid per component, but `aelf retire` and `aelf delete` both refuse a
locked belief without `--force`, so counting locks as removable prices
work the product will not do.

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
"""Shingles above this document frequency are store-wide boilerplate.

A belief posts to its shingles below the cap. Beliefs left with *no*
sub-cap shingle are not dropped — see `_blocked_pairs`, which rescues
them onto their rarest shared shingles instead. That fallback is the
#1316 fix; the cap itself is unchanged and still does the work of
keeping common phrasing from generating quadratic noise.
"""

DEFAULT_MAX_CANDIDATE_PAIRS: Final[int] = 2_000_000
"""Ceiling on candidate pairs *attempted* before the pass reports
`truncated`.

Bounds the one pathological shape blocking still admits: a corpus of
near-identical boilerplate, where a single bucket is quadratic in its own
size. Counting attempts rather than distinct pairs is what makes the
bound real — buckets re-proposing an already-held pair do not grow the
result set, so a size-based guard can never fire while the nested loop
still runs.

The development store attempts ~127k pairs at 44.5k active beliefs
(measured 2026-08-04, `aelf doctor --consolidate`), so this is ample
headroom rather than a working limit.
"""

MIN_COMPONENT_SIZE: Final[int] = 3
"""Smallest component the report counts. Pairs are plain supersession."""

MEDOID_SAMPLE_CAP: Final[int] = 64
"""Members over which the medoid is computed exactly.

The medoid needs every pairwise Levenshtein distance, which is O(K^2)
calls of an O(L^2) pure-Python routine. On the development store the two
largest components have 165 members each, and computing them exactly put
the whole pass at 74s — too slow for a `doctor` subcommand and, worse,
a function of near-duplicate density rather than of belief count, so it
degrades on exactly the corpus the report exists to describe.

Above this cap the medoid is chosen from the first `MEDOID_SAMPLE_CAP`
members in id order. Still an existing belief, still a closed-form
criterion, still deterministic — just evaluated over a bounded, named
subset.

This does **not** leave the counts untouched, and an earlier version of
this docstring claimed it did. `n_locked_in_clusters` excludes the
medoid, so on a component holding a locked member the price is a
function of *which* member is the medoid, and the cap can move that:
with a locked member in a 3-member component, cap 64 gives
`n_would_remove = 2` and cap 1 gives 1. Live exposure is small (3 locked
members across all clusters on the development store, and the shipped
cap reproduces the exact counts an uncapped pass gives there), but the
invariant is asserted rather than measured, so it is stated as the
approximation it is.
"""


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
    one medoid survives per component, and locked members are never
    retired, so it is
    `n_beliefs_in_clusters - n_clusters - n_locked_in_clusters`. It is
    the number that prices the intervention, which is why it is computed
    here rather than left to the caller to derive.

    `truncated` is True when the candidate-pair budget was reached, so
    every count below is a floor rather than a total.
    """

    n_beliefs_scanned: int
    n_candidate_pairs: int
    n_beliefs_rescued: int
    n_duplicate_pairs: int
    n_clusters: int
    n_beliefs_in_clusters: int
    n_locked_in_clusters: int
    largest_cluster: int
    jaccard_min: float
    levenshtein_min: float
    max_candidate_pairs: int
    max_shingle_df: int = DEFAULT_MAX_SHINGLE_DF
    truncated: bool = False
    clusters: tuple[ConsolidationCluster, ...] = field(default=())

    @property
    def n_would_remove(self) -> int:
        return max(
            0,
            self.n_beliefs_in_clusters
            - self.n_clusters
            - self.n_locked_in_clusters,
        )

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

    Bounded by `MEDOID_SAMPLE_CAP` — see that constant for why, and for
    what the approximation does and does not change.
    """
    pool = member_ids[:MEDOID_SAMPLE_CAP]
    best_id = pool[0]
    best_cost: int | None = None
    for candidate in pool:
        cost = sum(
            levenshtein_distance(content[candidate], content[other])
            for other in pool
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


def _blocked_pairs(
    shingle_sets: dict[str, frozenset[tuple[str, ...]]],
    max_candidate_pairs: int,
    max_shingle_df: int = DEFAULT_MAX_SHINGLE_DF,
) -> tuple[set[tuple[str, str]], bool, int]:
    """Candidate pairs from sub-cap shingles, plus a rescue for the rest.

    Returns `(candidates, truncated, n_rescued)`.

    Two posting passes, and the split is the whole point:

    * **Primary — every shared shingle with `df <= max_shingle_df`.**
      Identical to the pre-#1316 cap: a belief posts to all of its
      shingles that are not store-wide boilerplate. This pass alone
      reproduces the old candidate set exactly.
    * **Rescue — beliefs the primary pass leaves with no posting at
      all.** Those are the members of a family so large that *every*
      shingle they share is over the cap, which is #1316: the report
      announced zero duplicates precisely when the family was biggest
      (at the shipped cap of 32, a 32-member family reported 31
      removable and a 33-member family reported 0). They post to their
      rarest shared shingles instead, so a homogeneous family becomes
      its own bucket at any size.

    The first attempt at #1316 replaced the cap with rarest-shingle
    posting *for every belief*, which trades one cliff for another. A
    belief posts only to shingles tying its **own** minimum df, so two
    genuine near-duplicates with different minima never share a bucket
    and the family shatters into 2-components that `MIN_COMPONENT_SIZE`
    then discards — the same "reports 0 on the largest family" symptom,
    reached from the other side. Measured on the development store it
    dropped 490 beliefs and 81 whole clusters that the cap had found,
    including a 46-member clique in which all 1,035 pairs satisfy the
    shipped predicate. Making the rescue a *fallback* rather than a
    replacement is what keeps the fix strictly additive: every bucket
    the cap formed still forms, so the candidate set is a superset of
    the old one and no cluster can be lost.

    Buckets are consumed smallest-first — primary before rescue — so
    hitting the budget drops the largest, least discriminating ones.
    """
    df: dict[tuple[str, ...], int] = {}
    for shingles_of in shingle_sets.values():
        for gram in shingles_of:
            df[gram] = df.get(gram, 0) + 1

    postings: dict[tuple[str, ...], list[str]] = {}
    rescue: dict[tuple[str, ...], list[str]] = {}
    n_rescued = 0
    for bid, shingles_of in shingle_sets.items():
        # A df==1 shingle is unique to this belief and can never produce a
        # pair, so it is not a posting anywhere and must not define
        # "rarest" either — otherwise a belief with any unique tail (an
        # id, a version, a counter) posts only to shingles nothing else
        # shares and silently drops out.
        shared = [gram for gram in shingles_of if df[gram] >= 2]
        if not shared:
            continue
        under_cap = [gram for gram in shared if df[gram] <= max_shingle_df]
        if under_cap:
            for gram in under_cap:
                postings.setdefault(gram, []).append(bid)
            continue
        n_rescued += 1
        # Every shared shingle, not just the rarest ones. Posting a
        # rescued belief to its minimum-df shingles only would rebuild
        # the shattering this fallback exists to avoid one level down:
        # two rescued near-duplicates whose minima differ (one has a
        # df-35 shingle, the other bottoms out at the df-40 body) would
        # land in different buckets and the family would fragment into
        # components too small to report. The cost stays bounded because
        # a rescue bucket only ever holds *rescued* beliefs, which are
        # by construction the ones no sub-cap shingle reached.
        for gram in shared:
            rescue.setdefault(gram, []).append(bid)

    candidates: set[tuple[str, str]] = set()
    # The budget counts pairs *attempted*, not the distinct pairs kept.
    # `candidates` is a set, so buckets re-proposing a pair it already
    # holds do not grow it — a size-based guard can therefore stay false
    # while the nested loop runs Theta(S*N^2) times, which makes the
    # documented bound false for exactly the boilerplate shape it names.
    attempted = 0
    for source in (postings, rescue):
        buckets = sorted(source.values(), key=lambda m: (len(m), m[0]))
        for members in buckets:
            if len(members) < 2:
                continue
            members.sort()
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if attempted >= max_candidate_pairs:
                        return candidates, True, n_rescued
                    attempted += 1
                    candidates.add((members[i], members[j]))
    return candidates, False, n_rescued


def consolidation_audit(
    store: MemoryStore,
    *,
    jaccard_min: float = DEFAULT_JACCARD_MIN,
    levenshtein_min: float = DEFAULT_LEVENSHTEIN_MIN,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
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
    if max_candidate_pairs < 1:
        raise ValueError(
            f"max_candidate_pairs must be >= 1, got {max_candidate_pairs}"
        )
    # df 1 is unique-to-one-belief and can never pair, so a cap below 2
    # would post nothing through the primary pass and silently route the
    # entire store through the rescue fallback.
    if max_shingle_df < 2:
        raise ValueError(
            f"max_shingle_df must be >= 2, got {max_shingle_df}"
        )

    rows = store.list_beliefs_for_indexing()
    content = {bid: (text or "") for bid, text in rows}
    locked_ids = {b.id for b in store.list_locked_beliefs()}

    empty = ConsolidationReport(
        n_beliefs_scanned=len(rows),
        n_candidate_pairs=0,
        n_beliefs_rescued=0,
        n_duplicate_pairs=0,
        n_clusters=0,
        n_beliefs_in_clusters=0,
        n_locked_in_clusters=0,
        largest_cluster=0,
        jaccard_min=jaccard_min,
        levenshtein_min=levenshtein_min,
        max_candidate_pairs=max_candidate_pairs,
    )
    # Two beliefs are enough to form a *pair*, which the report counts
    # even though it takes three to form a cluster. Guarding on
    # MIN_COMPONENT_SIZE here would report `n_duplicate_pairs=0` on a
    # two-belief store that plainly has one.
    if len(rows) < 2:
        return empty

    tokens = {bid: tokenize(text) for bid, text in content.items()}
    token_sets = {bid: frozenset(toks) for bid, toks in tokens.items()}
    shingle_sets = {bid: shingles(toks) for bid, toks in tokens.items()}

    candidates, truncated, n_rescued = _blocked_pairs(
        shingle_sets, max_candidate_pairs, max_shingle_df
    )

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
    # A locked medoid is the survivor anyway, so only locked NON-medoid
    # members inflate the price.
    n_locked = sum(
        1
        for c in clusters
        for m in c.member_ids
        if m in locked_ids and m != c.medoid_id
    )

    return ConsolidationReport(
        n_beliefs_scanned=len(rows),
        n_candidate_pairs=len(candidates),
        n_beliefs_rescued=n_rescued,
        n_duplicate_pairs=len(duplicates),
        n_clusters=len(clusters),
        n_beliefs_in_clusters=in_clusters,
        n_locked_in_clusters=n_locked,
        largest_cluster=max((c.size for c in clusters), default=0),
        jaccard_min=jaccard_min,
        levenshtein_min=levenshtein_min,
        max_candidate_pairs=max_candidate_pairs,
        max_shingle_df=max_shingle_df,
        truncated=truncated,
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
        f"  candidate pairs       : {report.n_candidate_pairs:,}"
        + (
            f"  (BUDGET REACHED at {report.max_candidate_pairs:,} — every "
            "count below is a floor)"
            if report.truncated
            else ""
        ),
        f"  rescued past df={report.max_shingle_df:<4d} : "
        f"{report.n_beliefs_rescued:,} beliefs blocked on rarest shingles",
        f"  duplicate pairs       : {report.n_duplicate_pairs:,}",
        f"  clusters (size >= {MIN_COMPONENT_SIZE})  : {report.n_clusters:,}",
        f"  beliefs in a cluster  : {report.n_beliefs_in_clusters:,}",
        # Locked *medoids* are the survivor anyway, so they are neither
        # counted here nor priced; naming the line "user-locked" alone
        # over-claimed, since a locked medoid is silently absent from it.
        f"  locked non-medoids    : {report.n_locked_in_clusters:,} "
        "(never retired, so not priced)",
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
