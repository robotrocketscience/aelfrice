"""Reader-dependent and reader-independent metrics stay separated (#1160).

`aelf bench all` runs no reader, so the canonical adapters score the
joined retrieval context as though it were a model's answer. Two
consequences these tests pin:

- Metrics that are structurally uncomputable that way report the
  not-applicable sentinel instead of 0.0, and do not contribute to any
  aggregate.
- Rank metrics over the retrieved list are reported alongside, under
  `retrieval_quality`, so a ranking change has somewhere to show up that
  the token budget does not dominate.

The optional-dependency stubbing mirrors `tests/test_bench_adapters_
exit_code.py`: the dev/test environment installs only the core extras.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from benchmarks.metric_status import (
    NOT_APPLICABLE,
    NOT_APPLICABLE_REASONS_KEY,
)

# Importing the adapter pulls `aelfrice.retrieval`, and with it a cold
# scipy import that alone can exceed the 5 s global timeout in
# `pyproject.toml`. Whichever test imports first pays that cost, so the
# override has to cover the module rather than one named test. The
# assertions here are in-memory and finish in milliseconds; 30 s still
# catches a genuine hang.
pytestmark = pytest.mark.timeout(30)


def _stub_optional_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("nltk", "nltk.stem"):
        if name in sys.modules:
            continue
        stub = types.ModuleType(name)
        if name == "nltk.stem":
            class _PorterStemmer:
                def stem(self, w: str) -> str:
                    return w
            stub.PorterStemmer = _PorterStemmer  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, stub)


@pytest.fixture()
def locomo(monkeypatch: pytest.MonkeyPatch) -> Any:
    _stub_optional_deps(monkeypatch)
    import importlib
    return importlib.import_module("benchmarks.locomo_adapter")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _result_with(locomo: Any, scored: dict[int, tuple[int, float]],
                 unscorable_n: int) -> Any:
    """Build a BenchmarkResult without running a benchmark.

    `scored` maps category -> (question count, per-question F1).
    """
    r = locomo.BenchmarkResult()
    for cat, (n, f1) in scored.items():
        r.category_counts[cat] = n
        r.category_scores[cat] = [f1] * n
        r.total_qa += n
        r.total_f1 += f1 * n
    for cat in sorted(locomo.UNSCORABLE_CATEGORIES):
        r.category_counts[cat] = unscorable_n
        r.category_scores[cat] = []
        r.total_qa += unscorable_n
    return r


def test_overall_f1_excludes_the_unscorable_categories(locomo: Any):
    """The distinguishing assert: the old divisor was `total_qa`.

    With 20 scored questions at 0.5 and 5 unscorable ones, dividing by
    the full 25 yields 0.4 — a number that mixes ten real measurements
    with five placeholders. Dividing by the 20 that were scored yields
    the 0.5 that was actually measured.
    """
    r = _result_with(locomo, {1: (10, 0.5), 2: (10, 0.5)}, unscorable_n=5)
    assert r.total_qa == 25
    assert r.scored_qa == 20
    assert r.overall_f1 == pytest.approx(0.5)
    assert r.total_f1 / r.total_qa == pytest.approx(0.4)  # the old value


def test_total_qa_still_counts_every_question(locomo: Any):
    """`total_qa` is a corpus invariant the band-check watches for drift."""
    r = _result_with(locomo, {1: (10, 0.5)}, unscorable_n=5)
    assert r.total_qa == 15


def test_overall_f1_is_zero_when_nothing_was_scorable(locomo: Any):
    r = _result_with(locomo, {}, unscorable_n=5)
    assert r.scored_qa == 0
    assert r.overall_f1 == 0.0


def test_merge_carries_the_per_query_rank_metrics(locomo: Any):
    a = locomo.BenchmarkResult()
    a.per_question_retrieval.append({"reciprocal_rank": 1.0, "recall_at_1": 1.0})
    b = locomo.BenchmarkResult()
    b.per_question_retrieval.append({"reciprocal_rank": 0.0, "recall_at_1": 0.0})
    merged = locomo.merge_results([a, b])
    assert len(merged.per_question_retrieval) == 2
    assert merged.retrieval_quality()["mrr"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Emitted report
# ---------------------------------------------------------------------------


def _one_conversation(locomo: Any) -> Any:
    """A two-turn conversation with one scorable and one adversarial QA."""
    turns = [
        locomo.Turn(speaker="Alice", dia_id="D1:1",
                    text="I fly out of SFO whenever I visit my sister."),
        locomo.Turn(speaker="Bob", dia_id="D1:2",
                    text="Nice, I always end up at OAK instead."),
    ]
    session = locomo.LoCoMoSession(
        session_num=1, date_time="1:00 pm on 8 May, 2023", turns=turns,
    )
    return locomo.LoCoMoConversation(
        sample_id="synthetic_1", speaker_a="Alice", speaker_b="Bob",
        sessions=[session],
        qa_pairs=[
            locomo.QAPair(question="Which airport does Alice fly out of?",
                          answer="SFO", adversarial_answer="",
                          evidence=["D1:1"], category=4),
            locomo.QAPair(question="What is Alice's shoe size?",
                          answer="", adversarial_answer="Not mentioned",
                          evidence=[], category=5),
        ],
    )


@pytest.fixture()
def report(locomo: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    out = tmp_path / "locomo.json"
    monkeypatch.setattr(
        locomo, "load_locomo", lambda _p: [_one_conversation(locomo)],
    )
    monkeypatch.setattr(
        sys, "argv",
        ["locomo_adapter", "--data", str(tmp_path / "unused.json"),
         "--output", str(out)],
    )
    locomo.main()
    return json.loads(out.read_text())


def test_adversarial_category_reports_na_not_zero(report: dict[str, Any]):
    """0.0 is the worst possible score for a question never scored."""
    assert report["category_f1"]["5"] == NOT_APPLICABLE


def test_scorable_category_still_reports_a_number(report: dict[str, Any]):
    assert isinstance(report["category_f1"]["4"], float)


def test_report_says_why_the_category_is_na(report: dict[str, Any]):
    reasons = report[NOT_APPLICABLE_REASONS_KEY]
    assert "category_f1.5" in reasons
    assert "reader" in reasons["category_f1.5"]


def test_report_separates_scored_from_total(report: dict[str, Any]):
    assert report["total_qa"] == 2
    assert report["scored_qa"] == 1


def test_report_carries_reader_independent_retrieval_quality(report: dict[str, Any]):
    rq = report["retrieval_quality"]
    assert set(rq) >= {"mrr", "recall_at_1", "recall_at_5"}
    assert all(isinstance(v, float) for v in rq.values())


def test_retrieval_quality_covers_the_unscorable_questions_too(report: dict[str, Any]):
    """Whether the gold was retrieved is measurable without a reader.

    Both questions contribute, so a single question scoring rr=1.0 can
    only average to 0.5 — proof the adversarial row was not dropped.
    """
    assert len(report["per_question"]) == 2
    assert report["retrieval_quality"]["mrr"] <= 0.5


def test_per_question_carries_no_gold_derived_rank_metrics(report: dict[str, Any]):
    """`per_question` is the --retrieve-only reader payload.

    Rank metrics are computed against the gold answer, so leaking them
    into the reader's input would widen the surface
    `benchmarks/verify_clean.py` polices.
    """
    for row in report["per_question"]:
        assert "reciprocal_rank" not in row
        assert not any(k.startswith("recall_at_") for k in row)
