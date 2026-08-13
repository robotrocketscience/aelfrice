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
# scipy import. Whichever test imports first pays that cost, so the
# override has to cover the module rather than one named test. The
# assertions here are in-memory and finish in milliseconds; 30 s still
# catches a genuine hang.
#
# The marker no longer raises the budget — #1472 moved the base to the
# same 30 s — and it is kept deliberately. It states this module's own
# requirement, so it survives a future change to the base, and the
# import cost that motivated it is unmeasured, so replacing the old
# "can exceed 5 s" claim with a number for 30 s would be inventing
# one.
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


def test_retrieval_quality_excludes_the_unscorable_questions(report: dict[str, Any]):
    """The adversarial row must not dilute the rank metrics (#1160).

    Category 5 leaves `answer` empty, and `is_relevant` refuses an empty
    gold, so including it appends a 0.0 no ranking can move. The fixture
    has one scorable question retrieved at rank 1 and one adversarial
    question: `mrr` must read 1.0, not the 0.5 that averaging in the
    placeholder produces.

    This is the same defect the module exists to remove, so it is
    asserted rather than described — and it is asserted on the strict
    value, because `<= 0.5` would pass against the very behaviour being
    fixed.
    """
    assert len(report["per_question"]) == 2, "both questions still run"
    assert report["retrieval_quality"]["mrr"] == 1.0


def test_per_question_carries_no_gold_derived_rank_metrics(report: dict[str, Any]):
    """`per_question` is the --retrieve-only reader payload.

    Rank metrics are computed against the gold answer, so leaking them
    into the reader's input would widen the surface
    `benchmarks/verify_clean.py` polices.
    """
    for row in report["per_question"]:
        assert "reciprocal_rank" not in row
        assert not any(k.startswith("recall_at_") for k in row)


# ---------------------------------------------------------------------------
# exact_match across the blob-scoring adapters
# ---------------------------------------------------------------------------

# Every adapter in the canonical dispatcher that hands the joined
# retrieval context to an answer scorer. Exact match asks for whole-string
# equality against a short gold answer, which a multi-hundred-token
# context can never satisfy — so all four reported a structural 0.0.
_BLOB_SCORING_ADAPTERS = (
    "benchmarks.mab_adapter",
    "benchmarks.longmemeval_adapter",
    "benchmarks.amabench_adapter",
)


def _import_adapter(monkeypatch: pytest.MonkeyPatch, name: str) -> Any:
    import importlib
    _stub_optional_deps(monkeypatch)
    for dep in ("tiktoken", "datasets"):
        if dep in sys.modules:
            continue
        stub = types.ModuleType(dep)
        if dep == "tiktoken":
            class _Enc:
                def encode(self, s: str) -> list[int]:
                    return [0] * len(s)
            stub.encoding_for_model = lambda *a, **kw: _Enc()  # type: ignore[attr-defined]
            stub.get_encoding = lambda *a, **kw: _Enc()  # type: ignore[attr-defined]
        else:
            stub.load_dataset = lambda *a, **kw: []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, dep, stub)
    return importlib.import_module(name)


@pytest.mark.parametrize("module_name", _BLOB_SCORING_ADAPTERS)
def test_adapter_declares_exact_match_uncomputable(
    monkeypatch: pytest.MonkeyPatch, module_name: str,
):
    mod = _import_adapter(monkeypatch, module_name)
    assert "exact_match" in mod.UNCOMPUTABLE_METRICS
    assert "reader" in mod.UNCOMPUTABLE_METRICS["exact_match"]


@pytest.mark.parametrize("module_name", _BLOB_SCORING_ADAPTERS)
def test_adapter_exposes_reader_independent_retrieval_quality(
    monkeypatch: pytest.MonkeyPatch, module_name: str,
):
    """Every blob-scoring adapter reports rank metrics beside the F1."""
    mod = _import_adapter(monkeypatch, module_name)
    holder = {
        "benchmarks.mab_adapter": "MABResult",
        "benchmarks.longmemeval_adapter": "BenchmarkResult",
        "benchmarks.amabench_adapter": "AggregateResult",
    }[module_name]
    rq = getattr(mod, holder)().retrieval_quality()
    assert set(rq) >= {"mrr", "recall_at_1", "recall_at_5"}


def test_amabench_per_group_breakdown_does_not_rebuild_the_zero(
    monkeypatch: pytest.MonkeyPatch,
):
    """The distinguishing assert for `_accuracy_by_key`.

    An n/a metric contributes nothing to the group sum, so dividing by
    the row count reconstructs exactly the 0.0 the sentinel replaced —
    silently, and per group. Without the sentinel pass at the end of
    `_accuracy_by_key` this reads 0.0.
    """
    mod = _import_adapter(monkeypatch, "benchmarks.amabench_adapter")
    rows: list[dict[str, Any]] = [
        {"qa_type": "A", "exact_match": NOT_APPLICABLE,
         "substring_exact_match": 1.0, "f1": 0.5},
        {"qa_type": "A", "exact_match": NOT_APPLICABLE,
         "substring_exact_match": 0.0, "f1": 0.1},
    ]
    out = mod._accuracy_by_key(rows, "qa_type")
    assert out["A"]["exact_match"] == NOT_APPLICABLE
    assert out["A"]["substring_exact_match"] == pytest.approx(0.5)
    assert out["A"]["count"] == 2


# ---------------------------------------------------------------------------
# The shipped canonical file still band-checks
# ---------------------------------------------------------------------------

_CANONICAL = Path(__file__).resolve().parents[1] / "benchmarks/results/v2.0.0.json"


def _observed_in_the_new_shape(scale_overall_f1: float) -> dict[str, Any]:
    """The canonical report as the changed adapters would now emit it.

    `benchmarks/results/v2.0.0.json` was cut before this change and still
    records 0.0 for every metric now reported n/a. It cannot be recut
    from a unit test — the canonical cut needs the real datasets — so the
    coupling is exercised against the shipped file directly.
    """
    import copy

    from benchmarks import tolerance

    cano = tolerance.load_report(_CANONICAL)
    obs = copy.deepcopy(cano)
    results = obs["results"]
    for split in results["mab"]:
        results["mab"][split]["output"]["exact_match"] = NOT_APPLICABLE
    locomo_out = results["locomo"]["_"]["output"]
    locomo_out["category_f1"]["5"] = NOT_APPLICABLE
    # Dropping category 5 from the divisor raises overall_f1 by the share
    # the category held. The true share is not recorded in the canonical
    # file, so the caller sweeps it.
    locomo_out["overall_f1"] = round(locomo_out["overall_f1"] * scale_overall_f1, 4)
    return obs


@pytest.mark.parametrize("scale", [1.0, 1.1, 1.3, 2.0, 3.4])
def test_new_adapter_shape_never_fails_the_shipped_canonical(scale: float):
    """The change must not turn the nightly red before the recut lands.

    `overall_f1` can only rise, and it is higher-is-better, so leaving
    the band is a WARN on the improving side (#1209) rather than a FAIL.
    Every metric now reported n/a resolves to NOT_APPLICABLE rather than
    to the "not numeric" FAIL branch.
    """
    from benchmarks import tolerance

    cano = tolerance.load_report(_CANONICAL)
    checks = tolerance.check_report(cano, _observed_in_the_new_shape(scale))
    overall, counts = tolerance.summarize(checks)
    fails = [c for c in checks if c.verdict == tolerance.Verdict.FAIL]
    assert fails == [], [(c.path, c.note) for c in fails]
    assert overall in (tolerance.Verdict.PASS, tolerance.Verdict.WARN)
    # 4 MAB splits + LoCoMo category 5. LongMemEval and AMA-Bench also
    # report exact_match as n/a, but their canonical cut predates the
    # metric, so those leaves are not in the tree being walked.
    assert counts[tolerance.Verdict.NOT_APPLICABLE.value] == 5
    assert counts[tolerance.Verdict.PASS.value] > 0


# ---------------------------------------------------------------------------
# LongMemEval carries the same shape (#1160)
# ---------------------------------------------------------------------------


def test_longmemeval_abstention_questions_are_unscorable_for_retrieval(
    monkeypatch: pytest.MonkeyPatch,
):
    """The `_abs` set has no retrievable gold, so it must be excluded.

    Unlike LoCoMo category 5, these golds are long non-empty refusal
    sentences, so a guard on empty gold surfaces inside
    `retrieval_metrics` does **not** catch them — the unscorable set has
    to be declared per adapter. Measured on `longmemeval_s_cleaned.json`:
    0 of 30 `_abs` golds appear anywhere in their own haystack, against
    55 of 60 for a non-`_abs` control.
    """
    _stub_optional_deps(monkeypatch)
    import importlib

    lme = importlib.import_module("benchmarks.longmemeval_adapter")
    assert lme.is_unscorable_for_retrieval("abc_abs")
    assert not lme.is_unscorable_for_retrieval("abc")
    # A refusal-shaped gold is non-empty, so the empty-gold path cannot
    # be what protects the metric.
    from benchmarks.retrieval_metrics import is_relevant

    gold = [
        "You did not mention this information. "
        "You mentioned your cat Luna but not your hamster."
    ]
    assert gold[0].strip(), "gold is non-empty — an empty-gold guard would miss it"
    assert not is_relevant("I adopted a cat named Luna last spring.", gold)


def test_longmemeval_retrieval_quality_drops_the_abstention_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    """The aggregate must average only the scorable questions."""
    _stub_optional_deps(monkeypatch)
    import importlib

    lme = importlib.import_module("benchmarks.longmemeval_adapter")

    def _row(qid: str, rr: float) -> lme.RetrievalResult:
        return lme.RetrievalResult(
            question_id=qid,
            question_type="single-session-user",
            question="q",
            question_date="2023-01-01",
            answer="a",
            retrieved_context="ctx",
            num_beliefs=1,
            retrieval_latency_ms=0.0,
            rank_scores={"reciprocal_rank": rr, "recall_at_5": rr},
        )

    res = lme.BenchmarkResult()
    res.per_question.extend([_row("q1", 1.0), _row("q2_abs", 0.0)])
    rq = res.retrieval_quality()
    # `mean_metrics` reports `reciprocal_rank` under its conventional
    # aggregate name.
    assert rq["mrr"] == 1.0, (
        "the _abs row diluted the aggregate — it must be excluded"
    )
    assert rq["recall_at_5"] == 1.0
