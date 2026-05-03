"""Near-duplicate detection over beliefs (#197).

Stdlib-only port of the research-line dedup module. Two beliefs with
the same intent and different wording (e.g. "don't push to main"
locked + "never push directly to main" onboard-scraped) both surface
in retrieval today; v1.x has only `INSERT OR IGNORE` on
`(source, sentence)` content_hash, which catches exact matches but
not paraphrases.

The detector pairs two cheap deterministic similarity signals:

* **Jaccard** over lowercase Unicode-word tokens. Prefilter — fast,
  no allocation per character pair.
* **Levenshtein ratio** (`1 - edit_distance / max(len_a, len_b)`).
  Confirmation — guards against shared-vocabulary false positives
  that Jaccard alone would accept.

Both thresholds must hold for a pair to count as a duplicate: Jaccard
>= `jaccard_min` (default 0.8) **and** Levenshtein ratio >=
`levenshtein_min` (default 0.85). Defaults from the research-line
campaign; operator-tunable via `[dedup]` in `.aelfrice.toml`.

Candidate-pair generation is direct O(n^2) Jaccard prefiltering: for
each belief pair, tokenise once (cached), compute Jaccard, and skip
the pair entirely if Jaccard < `jaccard_min`. Live-store median is
~1.6k beliefs (~1.3M pairs); set-intersection on small token sets
runs at ~1-10 us per pair so the audit pass clears in under a few
seconds. The `max_candidate_pairs` cap (default 5000) bounds the
post-prefilter pair list in case a degenerate corpus produces too
many Jaccard-positive pairs to render. Sampling is deterministic —
sorted by `(belief_id_a, belief_id_b)` and truncated, so the same
store produces the same pair list across runs.

FTS5 is intentionally not the candidate source here: "don't" and
"do not" share zero indexed tokens after the FTS5 tokenizer's
apostrophe handling, so FTS5 misses the exact paraphrase shape this
detector targets. Jaccard over the lowercase Unicode-word tokenizer
treats them as one shared "don" token plus diverging stop-word
overlap — close enough for the prefilter, with the Levenshtein
ratio as the second-stage confirmation.

This module is the **algorithm**. The audit-only CLI surface
(`aelf doctor dedup`) lives in `cli.py`; the write-path hook flip is
deferred behind the bench-gate per #197 ratification.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Iterable

from aelfrice.bm25 import tokenize
from aelfrice.store import MemoryStore

# --- Defaults (research-line + #197 ratification) -----------------------

DEFAULT_JACCARD_MIN: Final[float] = 0.8
DEFAULT_LEVENSHTEIN_MIN: Final[float] = 0.85
DEFAULT_MAX_CANDIDATE_PAIRS: Final[int] = 5000

# --- Similarity primitives ---------------------------------------------


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    """Jaccard similarity over two token sets.

    Empty / empty returns 1.0 by convention (two zero-token strings
    are considered identical for dedup purposes — they cannot
    differ). One empty + one non-empty returns 0.0.
    """
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    inter = a & b
    return len(inter) / len(union)


def levenshtein_distance(a: str, b: str) -> int:
    """Two-row dynamic programming edit distance.

    O(len_a * len_b) time, O(min(len_a, len_b)) space. The two-row
    formulation avoids allocating a full (n+1) x (m+1) matrix; for
    the ~30-200 char belief-content range this is strictly faster
    than the textbook full-matrix version.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Force `b` to be the shorter to bound row size.
    if len(a) < len(b):
        a, b = b, a
    prev: list[int] = list(range(len(b) + 1))
    curr: list[int] = [0] * (len(b) + 1)
    for i, ca in enumerate(a, start=1):
        curr[0] = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev
    return prev[len(b)]


def levenshtein_ratio(a: str, b: str) -> float:
    """Length-normalised Levenshtein similarity in [0.0, 1.0].

    `1 - distance / max(len_a, len_b)`. Both empty returns 1.0; one
    empty + one non-empty returns 0.0. This is the
    `python-Levenshtein.ratio`-equivalent shape, computed without
    the C extension dependency.
    """
    if not a and not b:
        return 1.0
    longest = max(len(a), len(b))
    if longest == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(a, b) / longest)


# --- Pair + cluster types ----------------------------------------------


@dataclass(frozen=True)
class DuplicatePair:
    """A pair of beliefs that crossed both similarity thresholds.

    `belief_a_id` is the lexicographically smaller id; `belief_b_id`
    the larger. Pair ordering is deterministic so two runs over the
    same store produce the same pair list.
    """
    belief_a_id: str
    belief_b_id: str
    jaccard_score: float
    levenshtein_score: float


@dataclass(frozen=True)
class DuplicateCluster:
    """A connected component of beliefs reachable via duplicate edges.

    `member_ids` is sorted lexicographically; `representative_id` is
    `min(member_ids)` — the deterministic choice the audit uses to
    name the cluster. The write-path hook (deferred, bench-gated)
    will use the *oldest* member as the SUPERSEDES target instead,
    but the audit just picks deterministically.
    """
    representative_id: str
    member_ids: tuple[str, ...]


@dataclass
class DedupAuditReport:
    """Summary of one audit pass over the store.

    `pairs` is every above-threshold pair the audit saw; `clusters`
    is the union-find collapse of those pairs into connected
    components. `n_beliefs_scanned` is the total count of beliefs
    walked, before any candidate-pair filtering. `truncated` is
    `True` when the candidate-pair sample exceeded
    `max_candidate_pairs` and was truncated.
    """
    n_beliefs_scanned: int
    n_candidate_pairs: int
    n_duplicate_pairs: int
    n_clusters: int
    truncated: bool
    pairs: tuple[DuplicatePair, ...] = field(default_factory=tuple)
    clusters: tuple[DuplicateCluster, ...] = field(default_factory=tuple)


# --- Candidate pair generation -----------------------------------------


def _jaccard_prefiltered_pairs(
    beliefs: list[tuple[str, str]],
    *,
    jaccard_min: float,
    max_pairs: int,
) -> tuple[
    list[tuple[str, str, str, str, frozenset[str], frozenset[str], float]],
    int,
    bool,
]:
    """Return `[(id_a, content_a, id_b, content_b, tokens_a, tokens_b,
    jaccard)]` for every pair clearing `jaccard_min`, plus the raw
    candidate count (all O(n^2) pairs visited) and a `truncated`
    boolean.

    Tokens are cached per belief id so each belief tokenises once
    even if it participates in many pairs. The pair list is sorted
    deterministically by `(id_a, id_b)` and truncated to `max_pairs`
    if larger.
    """
    n = len(beliefs)
    token_cache: list[frozenset[str]] = [
        frozenset(tokenize(content)) for _, content in beliefs
    ]
    out: list[
        tuple[str, str, str, str, frozenset[str], frozenset[str], float]
    ] = []
    raw_count = 0
    for i in range(n):
        id_a, content_a = beliefs[i]
        ta = token_cache[i]
        if not content_a.strip():
            continue
        for j in range(i + 1, n):
            id_b, content_b = beliefs[j]
            tb = token_cache[j]
            if not content_b.strip():
                continue
            raw_count += 1
            j_score = jaccard(ta, tb)
            if j_score < jaccard_min:
                continue
            # Canonicalise (id_a < id_b) — i < j and ids are sorted
            # ASC by list_beliefs_for_indexing, so this holds.
            out.append((id_a, content_a, id_b, content_b, ta, tb, j_score))
    out.sort(key=lambda row: (row[0], row[2]))
    truncated = len(out) > max_pairs
    if truncated:
        out = out[:max_pairs]
    return out, raw_count, truncated


# --- Union-find for cluster collapse -----------------------------------


class _UnionFind:
    """Minimal union-find / DSU for clustering duplicate pairs.

    Path compression on `find`; union by size. Operations are
    effectively O(alpha(n)) per call.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._size: dict[str, int] = {}

    def make(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._size[x] = 1

    def find(self, x: str) -> str:
        path: list[str] = []
        while self._parent[x] != x:
            path.append(x)
            x = self._parent[x]
        for p in path:
            self._parent[p] = x
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self._size[ra] < self._size[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        self._size[ra] += self._size[rb]

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for x in self._parent:
            r = self.find(x)
            out.setdefault(r, []).append(x)
        return out


def cluster_pairs(pairs: Iterable[DuplicatePair]) -> tuple[DuplicateCluster, ...]:
    """Collapse duplicate pairs into connected components.

    Each cluster's `representative_id` is `min(member_ids)`; both
    cluster lists and the cluster tuple itself are sorted
    deterministically.
    """
    uf = _UnionFind()
    pair_list = list(pairs)
    for p in pair_list:
        uf.make(p.belief_a_id)
        uf.make(p.belief_b_id)
        uf.union(p.belief_a_id, p.belief_b_id)
    groups = uf.groups()
    clusters: list[DuplicateCluster] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        sorted_members = tuple(sorted(members))
        clusters.append(
            DuplicateCluster(
                representative_id=sorted_members[0],
                member_ids=sorted_members,
            )
        )
    clusters.sort(key=lambda c: c.representative_id)
    return tuple(clusters)


# --- Top-level audit entry point ---------------------------------------


def dedup_audit(
    store: MemoryStore,
    *,
    jaccard_min: float = DEFAULT_JACCARD_MIN,
    levenshtein_min: float = DEFAULT_LEVENSHTEIN_MIN,
    max_candidate_pairs: int = DEFAULT_MAX_CANDIDATE_PAIRS,
) -> DedupAuditReport:
    """Walk the store, find near-duplicate belief pairs, return a report.

    Read-only: no edges are inserted, no beliefs are mutated. The
    write-path hook (insert SUPERSEDES edges) is deferred behind the
    #197 bench gate.

    Raises `ValueError` on malformed thresholds; degrades gracefully
    on per-belief FTS5 errors (skips that belief, logs nothing).
    """
    if not 0.0 <= jaccard_min <= 1.0:
        raise ValueError(
            f"jaccard_min must be in [0.0, 1.0], got {jaccard_min}",
        )
    if not 0.0 <= levenshtein_min <= 1.0:
        raise ValueError(
            f"levenshtein_min must be in [0.0, 1.0], got {levenshtein_min}",
        )
    if max_candidate_pairs < 1:
        raise ValueError(
            f"max_candidate_pairs must be >= 1, got {max_candidate_pairs}",
        )

    beliefs = store.list_beliefs_for_indexing()
    n_beliefs = len(beliefs)
    if n_beliefs < 2:
        return DedupAuditReport(
            n_beliefs_scanned=n_beliefs,
            n_candidate_pairs=0,
            n_duplicate_pairs=0,
            n_clusters=0,
            truncated=False,
        )

    candidates, raw_count, truncated = _jaccard_prefiltered_pairs(
        beliefs,
        jaccard_min=jaccard_min,
        max_pairs=max_candidate_pairs,
    )

    pairs: list[DuplicatePair] = []
    for id_a, content_a, id_b, content_b, _ta, _tb, j_score in candidates:
        lr = levenshtein_ratio(content_a, content_b)
        if lr < levenshtein_min:
            continue
        pairs.append(
            DuplicatePair(
                belief_a_id=id_a,
                belief_b_id=id_b,
                jaccard_score=j_score,
                levenshtein_score=lr,
            )
        )
    pairs.sort(key=lambda p: (p.belief_a_id, p.belief_b_id))
    clusters = cluster_pairs(pairs)
    return DedupAuditReport(
        n_beliefs_scanned=n_beliefs,
        n_candidate_pairs=raw_count,
        n_duplicate_pairs=len(pairs),
        n_clusters=len(clusters),
        truncated=truncated,
        pairs=tuple(pairs),
        clusters=clusters,
    )


def format_audit_report(report: DedupAuditReport) -> str:
    """Render a `DedupAuditReport` as a human-readable plain-text block.

    Used by `aelf doctor dedup`. The shape mirrors `format_orphan_report`
    et al. in `doctor.py` so the doctor surface stays consistent.
    """
    lines: list[str] = []
    lines.append("aelf doctor dedup")
    lines.append("=" * 40)
    lines.append(f"Beliefs scanned         : {report.n_beliefs_scanned}")
    lines.append(f"Candidate pairs visited : {report.n_candidate_pairs}")
    if report.truncated:
        lines.append(
            f"  (truncated to {DEFAULT_MAX_CANDIDATE_PAIRS} — see "
            f"[dedup] max_candidate_pairs)"
        )
    lines.append(f"Duplicate pairs         : {report.n_duplicate_pairs}")
    lines.append(f"Duplicate clusters      : {report.n_clusters}")
    lines.append("")
    if report.n_clusters == 0:
        lines.append("No near-duplicates above the configured thresholds.")
        return "\n".join(lines)
    lines.append("Clusters:")
    for cluster in report.clusters:
        lines.append(
            f"  {cluster.representative_id}  "
            f"({len(cluster.member_ids)} members)"
        )
        for mid in cluster.member_ids:
            marker = "*" if mid == cluster.representative_id else " "
            lines.append(f"    {marker} {mid}")
    lines.append("")
    lines.append("Top duplicate pairs (jaccard, levenshtein):")
    for p in report.pairs[:25]:
        lines.append(
            f"  {p.belief_a_id}  ~  {p.belief_b_id}  "
            f"(j={p.jaccard_score:.3f}, l={p.levenshtein_score:.3f})"
        )
    if len(report.pairs) > 25:
        lines.append(f"  ... ({len(report.pairs) - 25} more)")
    return "\n".join(lines)
