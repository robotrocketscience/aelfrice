"""Bench gate for #819 labeled rerank-relevance corpus.

Tracker module. Four downstream consumers gate on this corpus:

- **#769** — flip `use_type_aware_compression` default-OFF → default-ON
  (A2 recall@k on labeled corpus, A4 fidelity on rebuild_logs).
- **#724** — raise `DEFAULT_CLUSTER_EDGE_FLOOR` 0.4 → 0.6 (re-bench on
  `multi_fact` + this corpus once v0_2 lands).
- **#800 R5 / #817** — flip `use_zeta_rerank` default-OFF → default-ON.
  Campaign-verdict R5 was deferred pending this corpus.
- Feature-ablation harness (lab `docs/feature-inventory.md` Layer 2).

Public CI skips when ``AELFRICE_CORPUS_ROOT`` is unset, per the
directory-of-origin rule. The corpus may carry **synthetic-only**
rows in the public tree once the labeling protocol is followed (see
``docs/feature-rerank-relevance-corpus.md``); real aelfrice-store
captures stay lab-side.

Expected row schema (also pinned in ``tests/test_corpus_schema.py``,
``tests/corpus/v2_0/README.md``):

    {
      "id": "row-id",
      "provenance": "...",
      "labeller_note": "...",
      "label": "graded",
      "query": "natural language query",
      "beliefs": [{"id": "...", "text": "..."}, ...],
      "gold_top_k": ["b-id-1", "b-id-2", ...],
      "k": 10,
      "gold_ordering": ["b-id-1", ...]  # optional, full preference order
    }

This file is the **smoke harness**: it confirms rows parse and the
shape constraints the downstream consumers rely on (``gold_top_k``
references belief ids present in ``beliefs``, ``gold_ordering`` when
present is a permutation-prefix of those same ids and contains every
``gold_top_k`` member). The downstream gates (recall@k, ζ vs γ rerank,
etc.) consume the same rows but live in their own bench files so the
smoke check stays independent of consumer-specific thresholds.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module


@pytest.mark.bench_gated
def test_rerank_relevance_corpus_smoke(
    aelfrice_corpus_root: Path,
) -> None:
    """Every row parses and references belief ids that exist in its pool."""
    rows = load_corpus_module(aelfrice_corpus_root, "rerank_relevance")
    assert rows, "rerank_relevance corpus produced zero rows"

    for row in rows:
        assert "id" in row, f"row missing id: {row}"
        rid = row["id"]
        assert "query" in row, f"row {rid} missing query"
        assert "beliefs" in row, f"row {rid} missing beliefs"
        assert "gold_top_k" in row, f"row {rid} missing gold_top_k"
        assert "k" in row, f"row {rid} missing k"

        assert isinstance(row["beliefs"], list) and row["beliefs"], (
            f"row {rid} beliefs must be non-empty list"
        )
        belief_ids = {b["id"] for b in row["beliefs"]}

        gold_top_k = row["gold_top_k"]
        assert isinstance(gold_top_k, list) and gold_top_k, (
            f"row {rid} gold_top_k must be non-empty list"
        )
        missing = set(gold_top_k) - belief_ids
        assert not missing, (
            f"row {rid} gold_top_k references belief ids not in pool: "
            f"{sorted(missing)}"
        )
        # Set semantics — duplicates within gold_top_k are a labelling
        # bug, not a shape question. Catch them at smoke time.
        assert len(set(gold_top_k)) == len(gold_top_k), (
            f"row {rid} gold_top_k contains duplicate ids: {gold_top_k}"
        )

        k = row["k"]
        assert isinstance(k, int) and not isinstance(k, bool) and k >= 1, (
            f"row {rid} k must be int >= 1, got {k!r}"
        )

        # Optional `gold_ordering` — when present, must contain every
        # `gold_top_k` member (full order extends the set) and reference
        # only ids in the belief pool. Order is significant; the consumer
        # uses it for rank-biased overlap measurements.
        ordering = row.get("gold_ordering")
        if ordering is not None:
            assert isinstance(ordering, list) and ordering, (
                f"row {rid} gold_ordering must be non-empty list when present"
            )
            ord_missing = set(ordering) - belief_ids
            assert not ord_missing, (
                f"row {rid} gold_ordering references belief ids not in "
                f"pool: {sorted(ord_missing)}"
            )
            assert len(set(ordering)) == len(ordering), (
                f"row {rid} gold_ordering contains duplicate ids: {ordering}"
            )
            gold_set = set(gold_top_k)
            assert gold_set.issubset(set(ordering)), (
                f"row {rid} gold_ordering must cover every gold_top_k id; "
                f"missing: {sorted(gold_set - set(ordering))}"
            )
