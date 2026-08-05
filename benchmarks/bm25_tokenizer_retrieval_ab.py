"""What the #1348 tokeniser change does to BM25F results (#1348, #1268).

`benchmarks/bm25_fts5_divergence.py` answers "does the index vocabulary
match FTS5". This answers the question that actually gates the change:
BM25F is the **default** retrieval lane — `resolve_use_bm25f_anchors`
returns True with no env, no kwarg and no TOML — so re-keying its
vocabulary moves production results. #1348 requires that movement be
reported rather than asserted.

Two arms over one store, differing only in the tokeniser the index is
built and queried with:

``legacy``
    `\\w+` + unguarded Porter — what shipped before #1348.
``shipped``
    `aelfrice.bm25.tokenize_stemmed` as it stands now.

The instrument is the label-free one the #1268 hold used, for the reason
it used it: there is no labelled retrieval corpus for this store, and a
gold set has not been funded. So this reports **agreement and reach**,
not quality — rank correlation and top-10 overlap between the arms, plus
the counts of queries that gain or lose a non-empty result set. A query
going from zero hits to some hits is the recall claim #1348 makes; a
query going the other way is the precision cost it concedes. Both are
counted, and neither is called an improvement here.

Queries are user turns recovered from the hook audit, filtered with
`r3_idf_clip_bound`'s own predicate — imported rather than re-spelled,
because #1268 made that filter mandatory and two copies would drift.
Note the audit stores `prompt_prefix`, which is truncated, so these are
prefixes of real turns rather than the turns themselves.

Usage::

    uv run python benchmarks/bm25_tokenizer_retrieval_ab.py \\
        --store .git/aelfrice/memory.db \\
        --audit .git/aelfrice/hook_audit.jsonl \\
                .git/aelfrice/hook_audit.jsonl.1
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.stats import kendalltau  # noqa: E402

import aelfrice.bm25 as bm25  # noqa: E402
from aelfrice.bm25 import BM25Index  # noqa: E402
from aelfrice.store import MemoryStore  # noqa: E402
from r3_idf_clip_bound import load_prompts  # noqa: E402

_LEGACY_PATTERN = re.compile(r"\w+", re.UNICODE)

TOP_K = 50
OVERLAP_AT = 10


def legacy_tokenize_stemmed(text: str) -> list[str]:
    """The pre-#1348 pipeline. Deliberately not reached through `bm25`."""
    if not text:
        return []
    return [
        bm25._PORTER_STEMMER.stemWord(m.group(0).lower())
        for m in _LEGACY_PATTERN.finditer(text)
    ]


def build_and_score(
    store: MemoryStore, queries: list[str], legacy: bool,
) -> list[list[tuple[str, float]]]:
    """Top-`TOP_K` hits per query, under one tokeniser.

    `BM25Index.build` and `.score` both resolve `tokenize_stemmed` as a
    module global, so swapping it here swaps the tokeniser on **both**
    the document and the query side — which is the point. Swapping only
    one would measure a bug nobody shipped.
    """
    original = bm25.tokenize_stemmed
    if legacy:
        bm25.tokenize_stemmed = legacy_tokenize_stemmed
    try:
        index = BM25Index.build(store)
        return [index.score(q, top_k=TOP_K) for q in queries]
    finally:
        bm25.tokenize_stemmed = original


def rank_agreement(
    left: list[tuple[str, float]], right: list[tuple[str, float]],
) -> float | None:
    """Kendall tau over the ids both arms returned.

    Restricted to the intersection on purpose: tau is undefined across
    disjoint item sets, and padding the missing side with a sentinel
    rank would manufacture agreement or disagreement that neither arm
    expressed. Queries with fewer than two shared ids return None and
    are reported as unscorable rather than folded in as zero.
    """
    lrank = {bid: i for i, (bid, _s) in enumerate(left)}
    rrank = {bid: i for i, (bid, _s) in enumerate(right)}
    shared = sorted(set(lrank) & set(rrank))
    if len(shared) < 2:
        return None
    tau = kendalltau(
        [lrank[b] for b in shared], [rrank[b] for b in shared],
    ).statistic
    return None if tau != tau else float(tau)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--audit", type=Path, nargs="+", required=True)
    parser.add_argument("--limit-queries", type=int, default=0)
    args = parser.parse_args(argv)

    queries = load_prompts(list(args.audit))
    if args.limit_queries:
        queries = queries[:args.limit_queries]
    if not queries:
        print("no user-turn prompts in the audit files", file=sys.stderr)
        return 2

    # #1328: read-only. A bare open runs migrations plus the #1314
    # lock-expiry sweep, and measuring a store is not a reason to write.
    store = MemoryStore(str(args.store), read_only=True)
    try:
        before = build_and_score(store, queries, legacy=True)
        after = build_and_score(store, queries, legacy=False)
    finally:
        store.close()

    n = len(queries)
    identical_top10 = 0
    jaccards: list[float] = []
    taus: list[float] = []
    unscorable_tau = 0
    gained, lost, both_empty = 0, 0, 0

    for lhs, rhs in zip(before, after, strict=True):
        lids = [b for b, _s in lhs[:OVERLAP_AT]]
        rids = [b for b, _s in rhs[:OVERLAP_AT]]
        if lids == rids:
            identical_top10 += 1
        union = set(lids) | set(rids)
        jaccards.append(
            1.0 if not union else len(set(lids) & set(rids)) / len(union)
        )
        tau = rank_agreement(lhs, rhs)
        if tau is None:
            unscorable_tau += 1
        else:
            taus.append(tau)
        if not lhs and not rhs:
            both_empty += 1
        elif not lhs and rhs:
            gained += 1
        elif lhs and not rhs:
            lost += 1

    def pct(x: int) -> str:
        return f"{x:5d}  ({100.0 * x / n:5.1f}%)"

    print(f"store          : {args.store}")
    print(f"queries        : {n}   (user turns, truncated prefixes)")
    print(f"top-K scored   : {TOP_K}, overlap measured at {OVERLAP_AT}")
    print()
    print("agreement between the arms")
    print(f"  identical top-{OVERLAP_AT}      {pct(identical_top10)}")
    mean_jaccard = (
        f"{statistics.fmean(jaccards):.4f}" if jaccards else "n/a"
    )
    print(f"  mean Jaccard@{OVERLAP_AT}      {mean_jaccard}")
    if taus:
        print(
            f"  median Kendall tau   {statistics.median(taus):.4f}"
            f"   (n={len(taus)}, unscorable={unscorable_tau})"
        )
    else:
        print(f"  median Kendall tau   n/a (unscorable={unscorable_tau})")
    print()
    print("reach — queries with a non-empty result set")
    print(f"  gained (0 -> hits)   {pct(gained)}")
    print(f"  lost   (hits -> 0)   {pct(lost)}")
    print(f"  empty in both arms   {pct(both_empty)}")
    print()
    print(
        "This is agreement and reach, not quality. Neither arm is scored\n"
        "against relevance labels, because no labelled corpus exists for\n"
        "this store — see the #1268 hold."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
