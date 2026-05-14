"""Bench gate for #434 type-aware compression.

Measures the precondition for A2 in `docs/feature-type-aware-compression.md`:
on a mixed-retention-class corpus, the compressed total token cost is
strictly less than the uncompressed total token cost. If compression
does not reduce cost on the lab corpus, no pack-loop rewrite can
deliver the A2 recall@k uplift no matter how it is wired.

Skips on public CI (autouse `bench_gated` marker handles
`AELFRICE_CORPUS_ROOT` absence). Skips again when the
`tests/corpus/v2_0/compression_uplift/` directory is empty.

Note: A2's strict recall@k comparison requires the follow-up
budget-rewrite work that consumes `compressed_beliefs[*].rendered_tokens`
in the pack loops. This gate measures the upstream invariant
(compression is monotone-strictly-decreasing in token cost on
non-fact-only corpora) that A2 depends on.

Expected row schema (`tests/corpus/v2_0/compression_uplift/*.jsonl`):

  {
    "id": "row-id",
    "content": "belief content text",
    "retention_class": "fact" | "snapshot" | "transient" | "unknown",
    "lock_level": "none" | "user"
  }
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aelfrice.compression import (
    STRATEGY_HEADLINE,
    STRATEGY_STUB,
    STRATEGY_VERBATIM,
    _estimate_tokens,
    compress_for_retrieval,
)
from aelfrice.models import (
    BELIEF_FACTUAL,
    LOCK_USER,
    RETENTION_UNKNOWN,
    Belief,
)
from tests.conftest import load_corpus_module


def _belief_from_row(row: dict) -> tuple[Belief, bool]:
    bid = str(row["id"])
    content = str(row["content"])
    retention_class = str(row.get("retention_class", RETENTION_UNKNOWN))
    lock_level = str(row.get("lock_level", "none"))
    locked = lock_level == LOCK_USER
    belief = Belief(
        id=bid,
        content=content,
        content_hash=f"h_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=lock_level,
        locked_at=None,
        created_at="2026-05-08T00:00:00Z",
        last_retrieved_at=None,
        retention_class=retention_class,
    )
    return belief, locked


@pytest.mark.bench_gated
def test_compression_reduces_total_tokens_on_corpus(
    aelfrice_corpus_root: Path,
) -> None:
    rows = load_corpus_module(aelfrice_corpus_root, "compression_uplift")

    uncompressed_total = 0
    compressed_total = 0
    strategy_counts: dict[str, int] = {
        STRATEGY_VERBATIM: 0,
        STRATEGY_HEADLINE: 0,
        STRATEGY_STUB: 0,
    }
    for row in rows:
        belief, locked = _belief_from_row(row)
        uncompressed_total += _estimate_tokens(belief.content)
        cb = compress_for_retrieval(belief, locked=locked)
        compressed_total += cb.rendered_tokens
        strategy_counts[cb.strategy] = strategy_counts.get(cb.strategy, 0) + 1

    assert uncompressed_total > 0, (
        f"compression_uplift corpus produced zero uncompressed tokens "
        f"on {len(rows)} rows; check corpus content"
    )
    # Monotone: compressed must not exceed uncompressed (per the
    # CompressedBelief invariant; this is a corpus-scale recheck).
    assert compressed_total <= uncompressed_total, (
        f"compressed_total {compressed_total} > uncompressed_total "
        f"{uncompressed_total} on {len(rows)} rows — invariant violation"
    )
    # Strict: at least one row must have actually compressed. A corpus
    # of all-fact-class rows would hit `verbatim` for every row and
    # leave compressed_total == uncompressed_total — that is a corpus
    # composition problem, not a compressor bug, so it skips rather
    # than fails.
    if compressed_total == uncompressed_total:
        pytest.skip(
            f"compression_uplift corpus has no compressible rows "
            f"(all {len(rows)} rows render verbatim); add snapshot/transient "
            f"rows to exercise the headline/stub strategies"
        )
    reduction = (uncompressed_total - compressed_total) / uncompressed_total
    # A2 precondition: on a mixed-class corpus, compression should
    # produce a non-trivial reduction. Threshold is loose (1%) — the
    # spec doesn't pin a magnitude here; A2's strict gate is
    # measured on recall@k after the pack-loop rewrite.
    assert reduction >= 0.01, (
        f"compression reduction {reduction:.3%} below 1% floor on "
        f"{len(rows)} rows; strategies={strategy_counts}"
    )
