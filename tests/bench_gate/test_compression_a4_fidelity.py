"""Bench gate for #769 / #775 type-aware compression — A4 rebuilder fidelity.

Spec § A4 in ``docs/feature-type-aware-compression.md``:

    continuation_fidelity(use_type_aware_compression=ON)
    >= continuation_fidelity(=OFF) - 0.005

at fixed ``[rebuilder] token_budget``, on a transcript-prefix +
expected-post-clear-answer corpus. The 0.005 tolerance band mirrors
the BM25F gate-band model at #154 (a half-point of the fidelity-score
noise floor).

Sibling to ``tests/bench_gate/test_compression_a2_recall.py``: A2 is
the retrieval-side gate (recall@k at fixed token_budget); A4 is the
rebuilder-side gate (continuation fidelity at fixed rebuilder
token_budget). #769's flip-default decision requires both axes to
clear.

Fidelity proxy. The runner uses a deterministic token-coverage proxy
in lieu of captured agent answers (see
``tests.retrieve_uplift_runner`` § A4 block comment). When the
lab corpus later carries ``captured_post_clear_answers_{off,on}``
arrays, the runner can swap to the #138 exact-method scorer without
touching this test's contract.

Public CI skips when ``AELFRICE_CORPUS_ROOT`` is unset, per the
directory-of-origin rule (labelled corpus rows live only in the
private companion repo at
``<corpus_root>/compression_a4_fidelity/*.jsonl`` — the public repo
carries only the schema contract and harness scaffold).

Expected row schema (`tests/corpus/v2_0/compression_a4_fidelity/*.jsonl`):

  {
    "id": "row-id",
    "transcript_pre_clear": [
      {"role": "user" | "assistant", "text": "...",
       "session_id": "...", "ts": "..."}
    ],
    "beliefs": [
      {"id": "...", "content": "...",
       "retention_class": "fact" | "snapshot" | "transient" | "unknown",
       "lock_level": "none" | "user"}
    ],
    "expected_post_clear_answers": ["answer-text-1", ...],
    "rebuilder_token_budget": 4000
  }

``rebuilder_token_budget`` is optional; falls back to
``DEFAULT_REBUILDER_TOKEN_BUDGET`` from
``aelfrice.context_rebuilder`` when absent.

Flip-default policy: this gate clears the A4 axis only. The full
flip-default decision for ``use_type_aware_compression=True`` also
requires A2 (retrieval recall@k uplift, covered by
``tests/bench_gate/test_compression_a2_recall.py``) and A3
(determinism, covered by ``tests/test_compression.py``). See
``docs/feature-type-aware-compression.md`` § Bench-gate / ship-or-defer
policy and #769 for the umbrella tracker.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import load_corpus_module


@pytest.mark.bench_gated
def test_compression_a4_corpus_round_trip(
    aelfrice_corpus_root: Path,
) -> None:
    """Smoke check: the compression_a4_fidelity corpus parses and every row
    exposes the spec § A4 fields. Skips when the directory is empty."""
    rows = load_corpus_module(aelfrice_corpus_root, "compression_a4_fidelity")
    assert rows, "compression_a4_fidelity corpus produced zero rows"

    for row in rows:
        assert "id" in row, f"row missing id: {row}"
        assert "transcript_pre_clear" in row, (
            f"row {row['id']} missing transcript_pre_clear"
        )
        assert "beliefs" in row, f"row {row['id']} missing beliefs"
        assert "expected_post_clear_answers" in row, (
            f"row {row['id']} missing expected_post_clear_answers"
        )
        assert isinstance(row["transcript_pre_clear"], list), (
            f"row {row['id']} transcript_pre_clear is not a list"
        )
        assert isinstance(row["beliefs"], list), (
            f"row {row['id']} beliefs is not a list"
        )
        assert isinstance(row["expected_post_clear_answers"], list), (
            f"row {row['id']} expected_post_clear_answers is not a list"
        )
        for i, turn in enumerate(row["transcript_pre_clear"]):
            assert "role" in turn, (
                f"row {row['id']} turn {i} missing role"
            )
            assert "text" in turn, (
                f"row {row['id']} turn {i} missing text"
            )
            assert turn["role"] in {"user", "assistant"}, (
                f"row {row['id']} turn {i} role must be user|assistant, "
                f"got {turn['role']!r}"
            )
        for i, b in enumerate(row["beliefs"]):
            assert "id" in b, f"row {row['id']} belief {i} missing id"
            assert "content" in b, (
                f"row {row['id']} belief {i} missing content"
            )
        for i, a in enumerate(row["expected_post_clear_answers"]):
            assert isinstance(a, str), (
                f"row {row['id']} expected_post_clear_answers[{i}] "
                f"must be a string, got {type(a).__name__}"
            )


@pytest.mark.bench_gated
@pytest.mark.timeout(120)
def test_compression_a4_fidelity_band(
    aelfrice_corpus_root: Path,
) -> None:
    """Spec § A4 ship-gate.

    Loads the ``compression_a4_fidelity`` corpus, drives every row through
    the OFF and ON rebuilder arms via ``run_compression_a4_fidelity``, and
    asserts the ON arm mean fidelity is at least the OFF arm mean fidelity
    minus the 0.005 tolerance band.
    """
    rows = load_corpus_module(aelfrice_corpus_root, "compression_a4_fidelity")
    assert rows, "compression_a4_fidelity corpus produced zero rows"

    runner_mod = pytest.importorskip(
        "tests.retrieve_uplift_runner",
        reason=(
            "compression A4 fidelity runner not available "
            "(operator gate; spec § A4 — pending lab-side corpus)"
        ),
    )

    results = runner_mod.run_compression_a4_fidelity(rows)
    tolerance = 0.005
    assert results.uplift >= -tolerance, (
        "type-aware compression must not regress continuation fidelity "
        f"by more than {tolerance:+.4f}\n"
        f"  n_rows  = {results.n_rows}\n"
        f"  OFF     = {results.mean_fidelity_off:.4f}\n"
        f"  ON      = {results.mean_fidelity_on:.4f}\n"
        f"  uplift  = {results.uplift:+.4f}\n"
        f"  band    = {tolerance:+.4f}"
    )
