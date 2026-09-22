"""Rank-based retrieval-quality metrics, independent of any reader.

`aelf bench all` runs no LLM. Every adapter in the canonical dispatcher
joins the retrieved beliefs into one string and scores that string as a
model answer, which makes the headline numbers a function of the token
budget rather than of retrieval quality: token-F1 between ~2000 tokens of
context and a three-token gold answer has precision ~3/2000, so *halving*
the budget roughly doubles the reported F1 while retrieving strictly less
(#1160).

The metrics here read the ranking instead of the blob. One retrieved
belief is counted relevant when a normalised gold surface occurs inside
it; `recall_at_k` asks whether such a belief reached the top k, and
`reciprocal_rank` asks how high the first one landed. Both are computed
over the ordered list the retriever returned, before it is joined.

That inverts the budget artifact. Token-F1 rises as the budget shrinks;
every metric here is monotone non-decreasing in the **number of items
kept**, so dropping items can only cost you. Improving the number
requires ranking a relevant belief higher, which is the thing the
benchmark is supposed to measure.
`tests/test_retrieval_metrics.py::test_keeping_fewer_items_never_raises_a_metric`
pins that property directly.

**These metrics are not monotone in the token budget, and this docstring
claimed they were** (#1574). The claim rested on "retrieval fills the
budget in rank order, so cutting the budget truncates the tail".
`clustering.pack_with_clusters` does not truncate: stage 1 abandons on
the first representative it cannot afford, and stage 2 skips an
over-budget belief and keeps filling. So the budget **selects** rather
than truncating, and a budget too small for a dear high-ranked belief
spends itself on cheaper lower-ranked ones. Raising it by one token
evicts them.

That reaches these metrics. With the gold-bearing belief cheap and
ranked second, one extra token of budget admits the dear irrelevant
belief, evicts the relevant one, and takes `recall_at_k` and
`reciprocal_rank` from 1.0 to 0.0 —
`tests/test_retrieval_metrics.py::test_raising_the_budget_can_lower_a_metric`
drives exactly that through production `retrieve()`.

Monotone in items kept does not give monotone in budget, because a
smaller budget does not keep a prefix of the same items. Read a
budget-to-budget comparison as a comparison of two different selections,
not of a list and its truncation. The skip-and-continue fill is
deliberate (#1574): the budget maximises how many beliefs are delivered,
and non-monotonicity is the accepted cost of that.

Multi-answer gold is treated as *alternative surfaces of one answer*, not
as several facts to be found, matching `qa_scoring.score_multi_answer`:
matching any listed surface counts as a hit. With a single relevant
answer, `recall_at_k` is the hit-rate@k of the IR literature.

Determinism: pure transformations over the inputs — no RNG, no clock, no
I/O — so two runs on identical inputs produce identical output.

Issue: #1160.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from benchmarks.qa_scoring import normalize_answer

#: Cut-offs reported by `retrieval_metrics`. 1 and 5 read the head of the
#: ranking, where a reader with a small context would look; 10 and 20 read
#: the depth a full budget actually injects.
DEFAULT_KS: Final[tuple[int, ...]] = (1, 5, 10, 20)


def is_relevant(item: str, ground_truths: Sequence[str]) -> bool:
    """True when any gold surface occurs inside `item` after normalisation.

    Empty gold surfaces are ignored rather than matched. `normalize_answer`
    strips punctuation and the articles a/an/the, so a gold answer of
    `"the"` normalises to `""` and `"" in anything` would otherwise award
    a free hit — the same defect `qa_scoring.score_substring_exact_match`
    guards against at its own entry point.
    """
    norm_item: str = normalize_answer(item)
    for gt in ground_truths:
        norm_gt: str = normalize_answer(gt)
        if norm_gt and norm_gt in norm_item:
            return True
    return False


def gold_ranks(
    retrieved: Sequence[str], ground_truths: Sequence[str],
) -> list[int]:
    """1-indexed ranks of every retrieved item that carries a gold surface.

    Rank 1 is the top of the retrieved list. An empty result means the
    gold answer is absent from the retrieved set entirely — which is the
    retrieval failure the blob scorers cannot distinguish from a reader
    failure.
    """
    return [
        rank
        for rank, item in enumerate(retrieved, start=1)
        if is_relevant(item, ground_truths)
    ]


def recall_at_k(
    retrieved: Sequence[str], ground_truths: Sequence[str], k: int,
) -> float:
    """1.0 when a gold-bearing item is within the top `k`, else 0.0.

    `k` below 1 always scores 0.0 — no prefix was inspected.
    """
    if k < 1:
        return 0.0
    return 1.0 if any(r <= k for r in gold_ranks(retrieved, ground_truths)) else 0.0


def reciprocal_rank(
    retrieved: Sequence[str], ground_truths: Sequence[str],
) -> float:
    """1/rank of the first gold-bearing item; 0.0 when there is none.

    Averaged over queries by `mean_metrics`, this is MRR.
    """
    ranks: list[int] = gold_ranks(retrieved, ground_truths)
    return 1.0 / ranks[0] if ranks else 0.0


def retrieval_metrics(
    retrieved: Sequence[str],
    ground_truths: Sequence[str],
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """Per-query reciprocal rank and recall at each cut-off in `ks`."""
    metrics: dict[str, float] = {
        "reciprocal_rank": reciprocal_rank(retrieved, ground_truths),
    }
    for k in ks:
        metrics[f"recall_at_{k}"] = recall_at_k(retrieved, ground_truths, k)
    return metrics


def mean_metrics(
    per_query: Sequence[dict[str, float]],
    *,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, float]:
    """Average per-query metrics into the aggregate block adapters report.

    `reciprocal_rank` averages into `mrr`, the conventional name.

    An empty input yields 0.0 for every key rather than omitting them, so
    the reported block has the same shape whether or not any query ran
    and a band-check never sees a leaf appear or vanish between runs.
    That 0.0 is a **shape placeholder, not a score** — it means "no query
    contributed", not "ranking was maximally bad". An empty run is
    already `NO_DATA` by other means, so nothing reads it as a
    measurement; the distinction is stated here because a 0.0 that looks
    like a score is the exact defect this module exists to remove.
    """
    keys: list[str] = ["reciprocal_rank", *(f"recall_at_{k}" for k in ks)]
    n: int = len(per_query)
    out: dict[str, float] = {}
    for key in keys:
        total: float = sum(float(q.get(key, 0.0)) for q in per_query)
        name: str = "mrr" if key == "reciprocal_rank" else key
        out[name] = round(total / n, 4) if n else 0.0
    return out
