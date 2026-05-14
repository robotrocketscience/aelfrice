"""Bench gate for #434 type-aware compression — A2 recall@k uplift.

Spec § A2 in ``docs/design/feature-type-aware-compression.md``:

    recall@k(use_type_aware_compression=ON) > recall@k(=OFF)

at fixed ``token_budget``, on a `(query, ground-truth, mixed-class
belief population)` triple corpus. Strict positive uplift is the
flip-default trigger for the A2 axis.

This gate is **distinct** from
``tests/bench_gate/test_compression_uplift.py`` which measures the
upstream invariant (compression reduces total tokens on a mixed-class
corpus). The upstream invariant is the precondition; A2 is the
recall@k measurement that the flip-default decision actually rides on.

Public CI skips when ``AELFRICE_CORPUS_ROOT`` is unset, per the
directory-of-origin rule (labelled corpus lives only in
``~/projects/aelfrice-lab/tests/corpus/v2_0/compression_a2_recall/``).

Expected row schema (`tests/corpus/v2_0/compression_a2_recall/*.jsonl`):

  {
    "id": "row-id",
    "query": "natural language query",
    "k": 12,
    "token_budget": 250,
    "beliefs": [
      {"id": "...", "content": "...",
       "retention_class": "fact" | "snapshot" | "transient" | "unknown",
       "lock_level": "none" | "user"}
    ],
    "expected_top_k": ["belief-id-1", ...]
  }

Flip-default policy: this gate clears the A2 axis only. The full
flip-default decision for ``use_type_aware_compression=True`` also
requires A3 (determinism, covered by ``tests/test_compression.py``)
and A4 (rebuilder continuation-fidelity, a separate bench gate
against the rebuild_logs corpus — not yet wired). See
``docs/design/feature-type-aware-compression.md`` § Bench-gate / ship-or-defer
policy.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module


@pytest.mark.bench_gated
def test_compression_a2_corpus_round_trip(
    aelfrice_corpus_root: Path,
) -> None:
    """Smoke check: the compression_a2_recall corpus parses and every row
    exposes the spec § A1 fields. Skips when the directory is empty."""
    rows = load_corpus_module(aelfrice_corpus_root, "compression_a2_recall")
    assert rows, "compression_a2_recall corpus produced zero rows"

    for row in rows:
        assert "id" in row, f"row missing id: {row}"
        assert "query" in row, f"row {row['id']} missing query"
        assert "expected_top_k" in row, f"row {row['id']} missing expected_top_k"
        assert "beliefs" in row, f"row {row['id']} missing beliefs"
        assert isinstance(row["beliefs"], list), (
            f"row {row['id']} beliefs is not a list"
        )
        assert isinstance(row["expected_top_k"], list), (
            f"row {row['id']} expected_top_k is not a list"
        )
        expected_set = set(row["expected_top_k"])
        belief_ids = {b["id"] for b in row["beliefs"]}
        missing = expected_set - belief_ids
        assert not missing, (
            f"row {row['id']} expected_top_k references "
            f"belief ids not in beliefs: {sorted(missing)}"
        )


@pytest.mark.bench_gated
@pytest.mark.timeout(60)
def test_compression_a2_strict_recall_uplift(
    aelfrice_corpus_root: Path,
) -> None:
    """Spec § A2 ship-gate.

    Loads the ``compression_a2_recall`` corpus, drives every row through
    the OFF and ON arms via ``run_compression_a2_uplift``, and asserts
    strictly positive mean recall@k uplift across the corpus.
    """
    rows = load_corpus_module(aelfrice_corpus_root, "compression_a2_recall")
    assert rows, "compression_a2_recall corpus produced zero rows"

    runner_mod = pytest.importorskip(
        "tests.retrieve_uplift_runner",
        reason=(
            "compression A2 uplift runner not available "
            "(operator gate; spec § A2 — pending lab-side corpus)"
        ),
    )

    results = runner_mod.run_compression_a2_uplift(rows)
    assert results.uplift > 0, (
        "type-aware compression must show strictly positive mean recall@k "
        "uplift across the compression_a2_recall corpus\n"
        f"  n_rows = {results.n_rows}\n"
        f"  OFF    = {results.mean_recall_off:.4f}\n"
        f"  ON     = {results.mean_recall_on:.4f}\n"
        f"  uplift = {results.uplift:+.4f}"
    )
