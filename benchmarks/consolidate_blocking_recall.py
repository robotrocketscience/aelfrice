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

Defaults to the repo-local ambient store.

**Read-only in the strong sense**: the store is opened as a `mode=ro`
SQLite connection rather than through `MemoryStore`, because constructing
a `MemoryStore` runs its open-time DDL, its pending one-shot migrations,
the `schema_meta` seed, and — since #1314 — the lock-expiry sweep, which
can flip a user's locks. A diagnostic must not mutate what it inspects,
and pointing this at the live store is the normal way to run it.

Exit status: 0 if the shipped blocking contains the flat cap, 1 if it
lost coverage, 2 if the store is missing, and 3 if the candidate-pair
budget bound (see `RESULT: INCONCLUSIVE` below).
"""
from __future__ import annotations

import sqlite3
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


def _read_active_beliefs(db: Path) -> list[tuple[str, str]]:
    """`[(belief_id, content)]` for active beliefs, id-ASC, without writing.

    Deliberately not `MemoryStore.list_beliefs_for_indexing` even though
    the query is copied from it: opening a `MemoryStore` runs migrations
    and the #1314 lock-expiry sweep, so the convenient call would mutate
    the store this script exists to measure. `mode=ro` makes that
    impossible rather than merely unlikely.
    """
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            "SELECT id, content FROM beliefs "
            "WHERE valid_to IS NULL ORDER BY id ASC"
        )
        return [(str(r[0]), str(r[1] or "")) for r in cur.fetchall()]
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    db = Path(argv[1]) if len(argv) > 1 else _DEFAULT_DB
    if not db.exists():
        print(f"no store at {db}", file=sys.stderr)
        return 2

    rows = _read_active_beliefs(db)
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

    contained = not lost_pairs and not lost_clusters and not (
        cap_members - ship_members
    )
    if truncated:
        # The reference arm is unbudgeted and this one is not, so once the
        # budget binds the two arms stop for different reasons and the
        # containment numbers say nothing about the fallback in either
        # direction: a non-empty `lost_pairs` means enumeration stopped,
        # and an empty one is equally uninformative. Reporting either as
        # PASS/FAIL would attribute a budget artefact to the mechanism,
        # and this script is cited as evidence for the superset claim.
        print(
            "RESULT: INCONCLUSIVE — the candidate-pair budget bound at "
            f"{DEFAULT_MAX_CANDIDATE_PAIRS:,}; raise it and re-run"
        )
        return 3
    print("RESULT: " + ("PASS — shipped blocking is a superset" if contained
                        else "FAIL — the fallback lost coverage"))
    return 0 if contained else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))
