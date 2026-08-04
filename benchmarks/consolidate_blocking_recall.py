"""#1316 blocking-recall audit — does the rescue fallback lose anything?

`consolidate._blocked_pairs` posts a belief to every shingle under
`max_shingle_df`, and rescues beliefs the cap leaves with **no** posting
onto their rarest shared shingles instead. The rescue exists because a
family larger than the cap shares no sub-cap shingle at all, so the audit
reported zero duplicates precisely when the family was biggest (#1316).

The safety claim the fix rests on is that the rescue is a **fallback**,
not a replacement: every bucket the flat cap formed still forms, so the
candidate set can only grow and no cluster the cap found can be lost.
That is a claim about a real store, not about a fixture, so this script
settles it against one rather than leaving it asserted in a docstring.

It matters because the first attempt at #1316 replaced the cap with
rarest-shingle posting for *every* belief, which reads as strictly better
and is not: two near-duplicates whose minimum `df` differs then never
share a bucket, families shatter into 2-components that
`MIN_COMPONENT_SIZE` discards, and the same "reports 0 on the largest
family" symptom returns from the other side. On the development store
that variant silently dropped 490 beliefs and 81 whole clusters. A
superset check is exactly the arm that would have caught it, so it ships
with the fix.

Transcribes the pre-#1316 flat-cap blocking inline (`_main_blocked`)
rather than importing it, because the shipped implementation is the fixed
one — there is nothing left in the tree to compare against.

Usage::

    uv run python benchmarks/consolidate_blocking_recall.py [PATH_TO_DB]

Defaults to the repo-local ambient store. Read-only: open the store
copy, never the live file, if anything may be writing to it. Exits
non-zero if any pair, cluster, or clustered belief present under the flat
cap is absent under the shipped blocking.
"""
from __future__ import annotations

import sys
from pathlib import Path

from aelfrice.bm25 import tokenize
from aelfrice.consolidate import (
    DEFAULT_MAX_CANDIDATE_PAIRS,
    DEFAULT_MAX_SHINGLE_DF,
    MIN_COMPONENT_SIZE,
    _blocked_pairs,
    _components,
    shingles,
)
from aelfrice.dedup import (
    DEFAULT_JACCARD_MIN,
    DEFAULT_LEVENSHTEIN_MIN,
    jaccard,
    levenshtein_ratio,
)
from aelfrice.store import MemoryStore

_DEFAULT_DB = Path(".git/aelfrice/memory.db")


def _main_blocked(
    shingle_sets: dict[str, frozenset[tuple[str, ...]]], cap: int
) -> set[tuple[str, str]]:
    """Pre-#1316 blocking: post everywhere, then skip over-cap buckets.

    Beliefs whose every shared shingle is over the cap contribute
    nothing — that is the defect, reproduced here so the comparison is
    against what actually shipped rather than against a description.
    """
    postings: dict[tuple[str, ...], list[str]] = {}
    for bid, shingles_of in shingle_sets.items():
        for gram in shingles_of:
            postings.setdefault(gram, []).append(bid)
    candidates: set[tuple[str, str]] = set()
    for members in postings.values():
        if len(members) < 2 or len(members) > cap:
            continue
        members.sort()
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidates.add((members[i], members[j]))
    return candidates


def main(argv: list[str]) -> int:
    db = Path(argv[1]) if len(argv) > 1 else _DEFAULT_DB
    if not db.exists():
        print(f"no store at {db}", file=sys.stderr)
        return 2

    store = MemoryStore(str(db))
    rows = store.list_beliefs_for_indexing()
    content = {bid: (text or "") for bid, text in rows}
    tokens = {bid: tokenize(text) for bid, text in content.items()}
    token_sets = {bid: frozenset(toks) for bid, toks in tokens.items()}
    shingle_sets = {bid: shingles(toks) for bid, toks in tokens.items()}
    print(f"store            : {db}")
    print(f"active beliefs   : {len(rows):,}")

    capped = _main_blocked(shingle_sets, DEFAULT_MAX_SHINGLE_DF)
    shipped, truncated, rescued = _blocked_pairs(
        shingle_sets, DEFAULT_MAX_CANDIDATE_PAIRS, DEFAULT_MAX_SHINGLE_DF
    )
    print(f"flat-cap pairs   : {len(capped):,}")
    print(
        f"shipped pairs    : {len(shipped):,} "
        f"(rescued {rescued:,} beliefs, truncated={truncated})"
    )

    def cluster(cands: set[tuple[str, str]]) -> list[list[str]]:
        dup = [
            (a, b)
            for a, b in sorted(cands)
            if jaccard(token_sets[a], token_sets[b]) >= DEFAULT_JACCARD_MIN
            and levenshtein_ratio(content[a], content[b])
            >= DEFAULT_LEVENSHTEIN_MIN
        ]
        return [
            g for g in _components(dup) if len(g) >= MIN_COMPONENT_SIZE
        ]

    cap_groups, ship_groups = cluster(capped), cluster(shipped)
    ship_sets = [set(g) for g in ship_groups]
    lost_pairs = capped - shipped
    lost_clusters = [
        g for g in cap_groups if not any(set(g) <= s for s in ship_sets)
    ]
    cap_members = {b for g in cap_groups for b in g}
    ship_members = {b for g in ship_groups for b in g}

    print(
        f"flat-cap clusters: {len(cap_groups)} "
        f"({len(cap_members)} beliefs, largest "
        f"{max((len(g) for g in cap_groups), default=0)})"
    )
    print(
        f"shipped clusters : {len(ship_groups)} "
        f"({len(ship_members)} beliefs, largest "
        f"{max((len(g) for g in ship_groups), default=0)})"
    )
    print(f"beliefs recovered: {len(ship_members - cap_members):,}")
    print("--- containment (all three must be 0) ---")
    print(f"pairs lost       : {len(lost_pairs):,}")
    print(f"clusters lost    : {len(lost_clusters):,}")
    print(f"beliefs lost     : {len(cap_members - ship_members):,}")

    ok = not lost_pairs and not lost_clusters and not (
        cap_members - ship_members
    )
    print("RESULT: " + ("PASS — shipped blocking is a superset" if ok
                        else "FAIL — the fallback lost coverage"))
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
