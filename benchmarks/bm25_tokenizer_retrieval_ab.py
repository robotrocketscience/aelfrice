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

**Two populations, and only one of them is evidence** (#1388).

``RECORDED``
    `input.extracted_query` from `.git/aelfrice/rebuild_logs/`.

    **This is not what production scores either, and the label says so
    rather than implying otherwise** (#1405). It is
    `_query_for_recent_turns(...)`, an entity/triple extraction, and
    neither caller hands that string to `retrieve()`:

    * `user_prompt_submit` — **97.7%** of recorded rows — scores
      `_build_conversation_aware_query(prompt, turns)`, the prompt
      repeated `conversation_aware_prompt_weight` times plus the last
      `turn_window` turns, composed in `hook.py` and never seen here.
      Default **on**.
    * `rebuild_v14` — 2.3% — scores `transform_query(raw_query, ...)`,
      which the log recomputes *before* the transform.

    So this arm is a third population. It is retained because it is the
    only recorded one, and because the shapes it does share with
    production (identifier density, length) are the ones a tokeniser
    change acts on. It is **not** the production number.

    Re-applying `transform_query` here was tried and reverted: it is
    correct for the 2.3% and wrong for the rest.

``DIAGNOSTIC``
    Raw user turns recovered from the hook audit, filtered with
    `r3_idf_clip_bound`'s own predicate — imported rather than
    re-spelled, because #1268 made that filter mandatory and two copies
    would drift. **Production never issues these.** The audit also
    stores `prompt_prefix`, which is truncated, so they are prefixes of
    real turns rather than the turns themselves.

Neither arm is the production population, and the honest state is that
**no arm can be, until `input.scored_query` (#1405) accumulates** — that
field records the string `retrieve()` was actually handed. Report both,
quote neither as "what production does", and revisit once the field has
data.

The two are still worth having together. A raw user turn is the input
*least* able to express a tokeniser change, because natural-language
turns rarely carry the identifier-shaped tokens the fix targets, so the
diagnostic arm systematically understates movement — which is how "movement
on real queries is small" came to be published. The recorded arm carries
extracted identifiers and moves more. Production, composing the prompt
with recent turns, plausibly sits between them; that is a hypothesis this
instrument cannot test.

**Emit aggregates only.** The production population is user prompt text.
Counts, shares and correlations may be printed or committed; query
strings, belief content and ids may not.

Usage::

    uv run python benchmarks/bm25_tokenizer_retrieval_ab.py \\
        --store .git/aelfrice/memory.db \\
        --audit .git/aelfrice/hook_audit.jsonl \\
                .git/aelfrice/hook_audit.jsonl.1

`--rebuild-logs` defaults to `<store parent>/rebuild_logs`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scipy.stats import kendalltau  # noqa: E402

# Imported as a module, not `from ... import`: `build_and_score` rebinds
# `bm25.tokenize_stemmed` to swap tokenisers, and a name bound at import
# would not see the swap.
import aelfrice.bm25 as bm25  # noqa: E402
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
        index = bm25.BM25Index.build(store)
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
    # NaN when one side has no rank variance at all; that is a
    # no-measurement, not an agreement of zero.
    return None if math.isnan(tau) else float(tau)


def load_production_queries(log_dir: Path) -> list[str]:
    """The strings production actually hands to `index.score` (#1388).

    `context_rebuilder` builds its query from a window of **user and
    assistant** turns and then rewrites it through `transform_query`; the
    result is what reaches `index.score`, and it is logged verbatim as
    `input.extracted_query` in the rebuild logs. So this population needs
    no reconstruction — it is a recording, not a model of one, which is
    the whole reason to prefer it over `_query_for_recent_turns` here.

    Deduplicated and sorted, so the population is a deterministic function
    of the log directory and two runs are comparable. Malformed lines are
    skipped rather than fatal: these logs are appended by a hook that can
    be killed mid-write.

    **Returns query text, which is user content.** Nothing derived from it
    may be printed or committed beyond aggregate counts — see the note in
    the module docstring.
    """
    seen: set[str] = set()
    for path in sorted(log_dir.glob("*.jsonl")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            payload = record.get("input")
            if not isinstance(payload, dict):
                continue
            query = payload.get("extracted_query")
            if isinstance(query, str) and query.strip():
                seen.add(query)
    return sorted(seen)


def report(
    label: str, caveat: str, store_label: str, queries: list[str],
    store: MemoryStore,
) -> dict[str, object]:
    """Score one population and print its block. Aggregates only."""
    before = build_and_score(store, queries, legacy=True)
    after = build_and_score(store, queries, legacy=False)

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

    print(f"=== {label} ===")
    print(f"store          : {store_label}")
    print(f"queries        : {n}   {caveat}")
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
    return {
        "population": label,
        "queries": n,
        "identical_top10": identical_top10,
        "identical_top10_share": round(identical_top10 / n, 4),
        "mean_jaccard_at_10": (
            round(statistics.fmean(jaccards), 4) if jaccards else None
        ),
        "median_kendall_tau": (
            round(statistics.median(taus), 4) if taus else None
        ),
        "tau_scored": len(taus),
        "tau_unscorable": unscorable_tau,
        "gained": gained,
        "lost": lost,
        "both_empty": both_empty,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--audit", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--rebuild-logs", type=Path, default=None,
        help=(
            "directory of rebuild_logs/*.jsonl carrying "
            "input.extracted_query — the population production issues "
            "(#1388). Defaults to <store parent>/rebuild_logs."
        ),
    )
    parser.add_argument("--limit-queries", type=int, default=0)
    args = parser.parse_args(argv)

    audit_queries = load_prompts(list(args.audit))
    log_dir = args.rebuild_logs or args.store.parent / "rebuild_logs"
    production_queries = (
        load_production_queries(log_dir) if log_dir.is_dir() else []
    )
    if args.limit_queries:
        audit_queries = audit_queries[:args.limit_queries]
        production_queries = production_queries[:args.limit_queries]
    if not audit_queries and not production_queries:
        print("no queries in either population", file=sys.stderr)
        return 2

    # #1328: read-only. A bare open runs migrations plus the #1314
    # lock-expiry sweep, and measuring a store is not a reason to write.
    store = MemoryStore(str(args.store), read_only=True)
    results: list[dict[str, object]] = []
    try:
        if production_queries:
            results.append(report(
                "RECORDED POPULATION — not what production scores",
                "(extracted_query as logged; see #1405)",
                args.store.name, production_queries, store,
            ))
        else:
            print(
                f"no rebuild logs under {log_dir} — the production arm is "
                "the one that answers #1388, and it did not run.",
                file=sys.stderr,
            )
        if audit_queries:
            results.append(report(
                "DIAGNOSTIC POPULATION — NOT the production input",
                "(raw user turns, truncated at the audit prefix cap)",
                args.store.name, audit_queries, store,
            ))
    finally:
        store.close()

    print(
        "This is agreement and reach, not quality. Neither arm is scored\n"
        "against relevance labels, because no labelled corpus exists for\n"
        "this store — see the #1268 hold.\n"
        "\n"
        "NEITHER ARM IS THE PRODUCTION POPULATION, and quoting one as if\n"
        "it were is what this benchmark got wrong twice. The recorded\n"
        "query is an entity extraction that no caller scores; the\n"
        "diagnostic arm is raw user turns, which the UPS path composes\n"
        "with recent turns before scoring. The field that settles it is\n"
        "`input.scored_query` (#1405) — revisit once it has data."
    )
    if len(results) == 2:
        prod, diag = results
        print(
            f"\nidentical top-10: recorded "
            f"{100 * float(prod['identical_top10_share']):.1f}% vs "
            f"diagnostic {100 * float(diag['identical_top10_share']):.1f}% "
            "— two approximations that bracket production, not a "
            "before/after."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
