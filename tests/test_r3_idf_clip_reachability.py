"""The R3 IDF-clip boost arm cannot fire against a real `BM25Index` (#1158 §4).

`tests/test_query_understanding.py` covers `clip_with_quantile_thresholds`
against hand-built `vocabulary` / `idf` pairs, where the boost arm fires
happily. It never composes the clip with the index that production actually
hands it, and the defect lives exactly in that composition:

* `compute_idf_quantile_thresholds` takes `high_threshold` as the 0.75
  quantile of the **vocabulary** IDF vector.
* The shipped IDF is Robertson's smoothed form,
  ``log(1 + (N - df + 0.5) / (df + 0.5))`` (`bm25.py`), strictly decreasing
  in `df`, so it attains its maximum at ``df == 1``.
* Natural-language vocabularies are Zipfian, so hapax terms are a large
  share of the vocabulary — far above the 25% needed to pull the 0.75
  quantile all the way up to that maximum.

When the quantile lands on the maximum, ``term_idf > high_threshold`` is
unsatisfiable for every term in the vocabulary, and `DEFAULT_BOOST_QF` is
dead code on the production path. Measured on the 44,593-belief development
store: hapax share 38.5%, ``high == max(idf) == 10.2999``, zero boostable
vocabulary terms, and zero boosts fired across 4,096 real query terms
(`benchmarks/r3_idf_clip_bound.py`).

This is the *composed* assertion the unit tests are missing. It is a
characterisation test: it pins current behaviour so that changing the
quantile policy — the actual fix — fails here loudly rather than silently.

Note this is the second independent reason the boost is inert. Even where it
fires, emitting `boost_qf` copies is a no-op in BM25F scoring at the shipped
``DEFAULT_K3 == 0.0``, per `tests/test_bm25_query_term_frequency.py` (#1166).
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from aelfrice.bm25 import BM25Index
from aelfrice.derivation import DerivationInput, derive
from aelfrice.models import INGEST_SOURCE_FILESYSTEM
from aelfrice.query_understanding.idf_clip import (
    DEFAULT_HIGH_QUANTILE,
    clip_with_quantile_thresholds,
    compute_idf_quantile_thresholds,
)
from aelfrice.store import MemoryStore
from benchmarks.r3_idf_clip_bound import load_prompts
from benchmarks.r3_idf_clip_bound import main as bound_main
from benchmarks.r3_idf_clip_bound import reachability

# A Zipfian-shaped corpus: every document shares a small common core and
# carries two document-unique terms. That puts the hapax share well above
# `1 - DEFAULT_HIGH_QUANTILE`, which is the only property the defect needs.
N_DOCS = 40


def _corpus() -> list[str]:
    return [
        f"the retrieval pipeline ranks beliefs and returns them "
        f"with marker{i} and token{i} attached"
        for i in range(N_DOCS)
    ]


@pytest.fixture
def index(tmp_path: Path) -> Iterator[BM25Index]:
    store = MemoryStore(str(tmp_path / "r3.db"))
    for i, text in enumerate(_corpus()):
        out = derive(
            DerivationInput(
                source_kind=INGEST_SOURCE_FILESYSTEM,
                raw_text=text,
                source_path=f"doc{i}.md",
                session_id=None,
                ts="2026-01-01T00:00:00+00:00",
            ),
        )
        assert out.belief is not None
        store.insert_or_corroborate(out.belief, source_type="filesystem_ingest")
    yield BM25Index.build(store)
    store.close()


def _hapax_share(index: BM25Index) -> float:
    idf_max = float(index.idf.max())
    return float((index.idf >= idf_max - 1e-6).sum()) / len(index.vocabulary)


def test_corpus_is_zipfian_enough_to_trigger_the_defect(
    index: BM25Index,
) -> None:
    """Precondition: hapax share must exceed `1 - DEFAULT_HIGH_QUANTILE`.

    If this fails the rest of the module proves nothing — the corpus, not
    the quantile policy, would be doing the work.
    """
    assert _hapax_share(index) > 1.0 - DEFAULT_HIGH_QUANTILE


def test_high_threshold_collapses_onto_max_idf(index: BM25Index) -> None:
    """The 0.75 quantile lands on the maximum attainable IDF."""
    _, high = compute_idf_quantile_thresholds(index.idf)
    assert high == pytest.approx(float(index.idf.max()))


def test_boost_arm_is_unreachable_on_a_real_index(index: BM25Index) -> None:
    """No vocabulary term satisfies the strict `idf > high` boost test."""
    _, high = compute_idf_quantile_thresholds(index.idf)
    assert int((index.idf > high).sum()) == 0


def test_clip_never_duplicates_any_vocabulary_term(index: BM25Index) -> None:
    """Behavioural consequence: the clip cannot emit a boosted copy.

    Feeding it the entire vocabulary is the most generous query possible;
    if the boost arm can fire at all, it fires here.
    """
    low, high = compute_idf_quantile_thresholds(index.idf)
    terms = sorted(index.vocabulary)
    out = clip_with_quantile_thresholds(
        terms, index.vocabulary, index.idf, low, high,
    )
    assert len(out) == len(set(out))


def test_a_reachable_high_quantile_does_boost(index: BM25Index) -> None:
    """Distinguishing arm: the collapse is the quantile's doing, not the corpus.

    Drop `high_quantile` below `1 - hapax_share` and the same corpus, the
    same index and the same clip function start emitting boosted copies. So
    the four assertions above measure the shipped 0.75 policy rather than
    restating something vacuously true of this fixture.
    """
    reachable_q = max(0.01, (1.0 - _hapax_share(index)) / 2.0)
    low, high = compute_idf_quantile_thresholds(
        index.idf, min(reachable_q / 2.0, 0.005), reachable_q,
    )
    assert int((index.idf > high).sum()) > 0

    terms = sorted(index.vocabulary)
    out = clip_with_quantile_thresholds(
        terms, index.vocabulary, index.idf, low, high,
    )
    assert len(out) > len(set(out))


def test_low_cutoff_is_reported_as_an_exact_document_frequency(
    index: BM25Index,
) -> None:
    """`df_at_low_cutoff` must invert the IDF the index actually uses.

    The harness ships to be re-run on other stores, so this has to hold away
    from the development store's operating point. Inverting
    ``log(1 + (N + 0.5) / (df + 0.5))`` — dropping the ``- df`` from the
    numerator of the shipped form — agrees to 0.02% where ``exp(low) >> 1``
    and is off by 58% at ``idf == 1.0``, so a smaller or less Zipfian corpus
    would be reported wrongly with no visible symptom.

    Feeding back the IDF of a known `df` must return that `df`. The dropped
    form fails this across the whole range on this fixture — 1.0380 against
    1.0 at df = 1, and 3280.0 against 40.0 at df = 40 — so the assertion
    measures the inversion rather than restating it. Note the error grows
    with `df`: it is mildest exactly where the development store's cutoff
    sits, which is why the defect survived a review that checked only there.
    """
    n_docs = len(index.belief_ids)
    idf_max = float(index.idf.max())
    for df in range(1, n_docs + 1):
        idf_at_df = float(np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)))
        reported = reachability(index, idf_at_df, idf_max)["df_at_low_cutoff"]
        assert reported == pytest.approx(df, rel=1e-9)


def test_low_cutoff_is_non_finite_exactly_when_the_cutoff_is_zero(
    index: BM25Index,
) -> None:
    """Pins the precondition `main`'s null-conversion exists for.

    A `low` of 0.0 admits every term, so there is no document frequency to
    report and the field is NaN. `main` depends on that being the only
    non-finite case when it serialises; this fails loudly if the
    representation changes to a sentinel number instead.
    """
    idf_max = float(index.idf.max())
    assert math.isnan(reachability(index, 0.0, idf_max)["df_at_low_cutoff"])
    assert math.isfinite(reachability(index, 1.0, idf_max)["df_at_low_cutoff"])


def test_json_out_is_parseable_by_a_strict_rfc_8259_reader(
    tmp_path: Path,
) -> None:
    """End-to-end: `--json-out` must never emit a bare `NaN` or `Infinity`.

    `json.dumps` writes those tokens by default and RFC 8259 does not permit
    them, so a strict parser rejects the file. `parse_constant` fires on
    exactly those three tokens, so this guards the whole payload against a
    future non-finite field, not `df_at_low_cutoff` alone.

    Scope, stated plainly: this does **not** exercise `main`'s NaN-to-null
    conversion, and would still pass with it deleted. Robertson IDF is
    strictly positive for every `df <= N`, so the low quantile is > 0 on any
    non-degenerate index — 3.3081 on this fixture — and `df_at_low_cutoff`
    is therefore finite on the whole reachable input range. That conversion
    is defensive rather than live, which is the same shape of finding as the
    unreachable boost arm this module exists to pin. The NaN branch itself
    is covered directly against `reachability` above.
    """
    store_path = tmp_path / "e2e.db"
    store = MemoryStore(str(store_path))
    for i, text in enumerate(_corpus()):
        out = derive(
            DerivationInput(
                source_kind=INGEST_SOURCE_FILESYSTEM,
                raw_text=text,
                source_path=f"doc{i}.md",
                session_id=None,
                ts="2026-01-01T00:00:00+00:00",
            ),
        )
        assert out.belief is not None
        store.insert_or_corroborate(out.belief, source_type="filesystem_ingest")
    store.close()

    audit = tmp_path / "hook_audit.jsonl"
    audit.write_text(
        "\n".join(
            json.dumps({"hook": "user_prompt_submit", "prompt_prefix": text})
            for text in _corpus()
        )
        + "\n",
        encoding="utf-8",
    )

    json_out = tmp_path / "bound.json"
    assert bound_main(
        [
            "--store", str(store_path),
            "--audit", str(audit),
            "--json-out", str(json_out),
        ],
    ) == 0

    def _reject(token: str) -> float:
        raise AssertionError(f"non-RFC-8259 token in --json-out: {token}")

    payload = json.loads(json_out.read_text(), parse_constant=_reject)
    assert "reachability" in payload


def test_a_missing_audit_path_warns_instead_of_measuring_a_partial_corpus(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A misspelt `--audit` path must not silently shrink the corpus.

    `load_prompts` takes several paths; skipping a bad one without a word
    yields numbers over whatever survived, with nothing in the output to say
    the measurement is partial.
    """
    good = tmp_path / "good.jsonl"
    good.write_text(
        json.dumps({"hook": "user_prompt_submit", "prompt_prefix": _corpus()[0]})
        + "\n",
        encoding="utf-8",
    )
    missing = tmp_path / "typo.jsonl"

    prompts = load_prompts([good, missing])

    assert len(prompts) == 1
    assert str(missing) in capsys.readouterr().err


def test_idf_is_monotone_decreasing_in_document_frequency(
    index: BM25Index,
) -> None:
    """The premise the reachability argument rests on, asserted directly.

    Robertson smoothed IDF is strictly decreasing in `df`, so `max(idf)` is
    attained exactly at `df == 1` and no term can ever exceed it.
    """
    n_docs = len(index.belief_ids)
    idf_at = [
        float(np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)))
        for df in range(1, min(n_docs, 12))
    ]
    assert idf_at == sorted(idf_at, reverse=True)
    assert float(index.idf.max()) == pytest.approx(idf_at[0], rel=1e-5)
