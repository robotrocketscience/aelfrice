"""A bench-gate corpus counts only when it defeats its own null model (#1581).

Two separate things are pinned here, because two separate ways of
losing the guard were observed:

* **Declaration.** Every module under `tests/bench_gate/` names a family
  and a null model in `GATE_DECLARATIONS`. A new gate that declares
  neither fails here rather than shipping as an unguarded "executed"
  count.
* **Behaviour.** One degenerate corpus per family, constructed to be
  degenerate in exactly that family's way, must be rejected; the
  healthy counterpart must be accepted. This is the mutation proof for
  the guard itself, run on every suite.

The corpora that exist live in the private lab repo, so the behavioural
tests here are built from synthetic rows. That is deliberate and not a
compromise: the states worth pinning include ones no corpus on this
machine can produce.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.bench_gate.null_model import (
    GATE_DECLARATIONS,
    AblationArms,
    Family,
    GateDeclaration,
    bar_at_least,
    guard_ablation_gate,
    guard_classification_gate,
    guard_ranking_gate,
    majority_constant_accuracy,
    precision_at_k,
    structural_prefilter,
)
from tests.conftest import (
    BENCH_MEASUREMENT_PROPERTY,
    BENCH_NULL_VERDICT_PROPERTY,
)

_BENCH_GATE_DIR = Path(__file__).resolve().parent / "bench_gate"


def _gate_files() -> list[Path]:
    return sorted(_BENCH_GATE_DIR.glob("test_*.py"))


class _Recorder:
    """Stands in for pytest's `record_property` fixture."""

    def __init__(self) -> None:
        self.properties: list[tuple[str, str]] = []

    def __call__(self, key: str, value: object) -> None:
        self.properties.append((key, str(value)))

    def value(self, key: str) -> str:
        matches = [v for k, v in self.properties if k == key]
        assert len(matches) == 1, f"expected one {key!r}, got {matches}"
        return matches[0]


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def test_every_bench_gate_module_declares_a_family_and_a_null_model() -> None:
    """AC1. A module that declares neither must not be able to ship."""
    undeclared = [p.name for p in _gate_files() if p.stem not in GATE_DECLARATIONS]
    assert not undeclared, (
        f"bench-gate modules with no null-model declaration: {undeclared}.\n"
        f"  Add an entry to GATE_DECLARATIONS in "
        f"tests/bench_gate/null_model.py naming the family and the null "
        f"model, or Family.EXEMPT with the reason no null model is "
        f"constructible from those rows. Per the #1581 ruling the check is a "
        f"precondition to counting, so an undeclared gate cannot be counted."
    )


def test_an_exempt_declaration_must_say_why() -> None:
    """An exemption is a declaration; a blank one is an omission."""
    with pytest.raises(ValueError, match="must give a reason"):
        GateDeclaration(family=Family.EXEMPT, null_model="none")


def test_a_scored_declaration_must_name_its_corpus_module() -> None:
    with pytest.raises(ValueError, match="must name its corpus module"):
        GateDeclaration(family=Family.RANKING, null_model="shuffle")


# ---------------------------------------------------------------------------
# Structural pre-filters
# ---------------------------------------------------------------------------


def _ranking_rows(
    n: int, *, pool: int, gold: int, k: int, gold_override: dict[int, int] | None = None
) -> list[dict]:
    rows = []
    for i in range(n):
        size = (gold_override or {}).get(i, gold)
        rows.append({
            "id": f"r{i}",
            "k": k,
            "beliefs": [{"id": f"r{i}-b{j}"} for j in range(pool)],
            "expected_top_k": [f"r{i}-b{j}" for j in range(size)],
        })
    return rows


def test_prefilter_rejects_a_gold_equals_pool_corpus() -> None:
    """The `query_strategy` shape: no distractor can be lost."""
    rejection = structural_prefilter(
        _ranking_rows(30, pool=10, gold=10, k=10),
        gold_key="expected_top_k",
        pool_key="beliefs",
    )
    assert rejection is not None
    assert rejection.name == "separability"
    assert "0.0%" in rejection.detail or "0%" in rejection.detail


def test_prefilter_rejects_one_row_carrying_the_corpus() -> None:
    """Micro-averaging turns a 30-row corpus into a 1-row corpus."""
    rows = _ranking_rows(30, pool=80, gold=2, k=5, gold_override={0: 60})
    rejection = structural_prefilter(
        rows, gold_key="expected_top_k", pool_key="beliefs"
    )
    assert rejection is not None
    assert rejection.name == "gold-skew"


def test_prefilter_accepts_a_separable_corpus() -> None:
    assert (
        structural_prefilter(
            _ranking_rows(30, pool=20, gold=3, k=5),
            gold_key="expected_top_k",
            pool_key="beliefs",
        )
        is None
    )


def test_prefilter_refuses_rows_it_cannot_read() -> None:
    """Silently passing on unreadable rows is the defect, not the fix."""
    with pytest.raises(KeyError, match="cannot read this corpus"):
        structural_prefilter(
            [{"id": "r0"}], gold_key="expected_top_k", pool_key="beliefs"
        )


# ---------------------------------------------------------------------------
# Behaviour, one degenerate corpus per family
# ---------------------------------------------------------------------------


def test_ranking_guard_rejects_a_corpus_a_shuffle_can_clear() -> None:
    """Separable rows, but so saturated that random order clears the bar."""
    rows = _ranking_rows(30, pool=11, gold=10, k=10)
    rec = _Recorder()
    with pytest.raises(pytest.fail.Exception, match="does not count"):
        guard_ranking_gate(
            module="synthetic_ranking",
            rows=rows,
            shipped=lambda: 0.95,
            bar=bar_at_least(0.5),
            metric=precision_at_k,
            record_property=rec,
            gold_key="expected_top_k",
            pool_key="beliefs",
        )
    assert "REJECT" in rec.value(BENCH_NULL_VERDICT_PROPERTY)


def test_ranking_guard_accepts_a_corpus_a_shuffle_cannot() -> None:
    rows = _ranking_rows(30, pool=20, gold=3, k=5)
    rec = _Recorder()
    result = guard_ranking_gate(
        module="synthetic_ranking",
        rows=rows,
        shipped=lambda: 0.80,
        bar=bar_at_least(0.5),
        metric=precision_at_k,
        record_property=rec,
        gold_key="expected_top_k",
        pool_key="beliefs",
    )
    assert result.null < 0.5 <= result.shipped
    assert "ACCEPT" in rec.value(BENCH_NULL_VERDICT_PROPERTY)


def test_a_green_gate_records_both_scores() -> None:
    """AC2's other half: a green run has to leave the evidence behind.

    Without this the release reviewer sees a pass and cannot tell a
    degenerate +1.000 from a real +0.06.
    """
    rec = _Recorder()
    guard_ranking_gate(
        module="synthetic_ranking",
        rows=_ranking_rows(30, pool=20, gold=3, k=5),
        shipped=lambda: 0.80,
        bar=bar_at_least(0.5),
        metric=precision_at_k,
        record_property=rec,
        gold_key="expected_top_k",
        pool_key="beliefs",
    )
    line = rec.value(BENCH_MEASUREMENT_PROPERTY)
    assert "shipped=0.8000" in line
    assert "null=" in line and "null_clears_bar=False" in line
    assert "verdict=ACCEPT" in line


def _labelled(counts: dict[str, int]) -> list[dict]:
    rows = []
    for label, n in counts.items():
        rows += [{"id": f"{label}-{i}", "label": label} for i in range(n)]
    return rows


def test_classification_guard_rejects_an_imbalanced_corpus() -> None:
    rows = _labelled({"negative": 27, "positive": 3})
    rec = _Recorder()
    with pytest.raises(pytest.fail.Exception, match="constant predictor"):
        guard_classification_gate(
            module="synthetic_classification",
            rows=rows,
            shipped=lambda: 0.93,
            bar=bar_at_least(0.5),
            score_constant=lambda label: majority_constant_accuracy(rows)[0],
            record_property=rec,
        )
    assert "REJECT" in rec.value(BENCH_NULL_VERDICT_PROPERTY)


def test_classification_guard_accepts_a_balanced_corpus() -> None:
    rows = _labelled({"positive": 11, "negative": 10, "neutral": 9})
    rec = _Recorder()
    result = guard_classification_gate(
        module="synthetic_classification",
        rows=rows,
        shipped=lambda: 0.71,
        bar=bar_at_least(0.5),
        score_constant=lambda label: majority_constant_accuracy(rows)[0],
        record_property=rec,
    )
    assert result.null < 0.5
    assert "ACCEPT" in rec.value(BENCH_NULL_VERDICT_PROPERTY)


def test_classification_guard_refuses_a_ranking_shaped_corpus() -> None:
    """`gold_key=None` is a claim about the rows, not a switch."""
    rows = _ranking_rows(3, pool=4, gold=1, k=2)
    for row in rows:
        row["label"] = "graded"
    with pytest.raises(AssertionError, match="declared no candidate pool"):
        guard_classification_gate(
            module="synthetic_classification",
            rows=rows,
            shipped=lambda: 1.0,
            bar=bar_at_least(0.5),
            score_constant=lambda label: 0.1,
            record_property=_Recorder(),
        )


def test_ablation_guard_rejects_a_feature_only_path_corpus() -> None:
    """Uplift +1.000 proves the weight table, not the feature."""
    rec = _Recorder()
    arms = AblationArms(shipped=1.0, ablated=0.0, without_row_scores=[0.0] * 30)
    with pytest.raises(pytest.fail.Exception, match="only path to the gold"):
        guard_ablation_gate(
            module="synthetic_ablation",
            rows=[{"id": f"r{i}"} for i in range(30)],
            arms=lambda: arms,
            bar=bar_at_least(0.05),
            record_property=rec,
        )
    assert "REJECT" in rec.value(BENCH_NULL_VERDICT_PROPERTY)


def test_ablation_guard_accepts_a_corpus_with_a_reachable_gold() -> None:
    rec = _Recorder()
    arms = AblationArms(
        shipped=0.61, ablated=0.48, without_row_scores=[1.0] * 14 + [0.0] * 16
    )
    result = guard_ablation_gate(
        module="synthetic_ablation",
        rows=[{"id": f"r{i}"} for i in range(30)],
        arms=lambda: arms,
        bar=bar_at_least(0.05),
        record_property=rec,
    )
    assert result.shipped == pytest.approx(0.61)
    assert "ACCEPT" in rec.value(BENCH_NULL_VERDICT_PROPERTY)
    assert "without_rate_share=0.467" in rec.value(BENCH_MEASUREMENT_PROPERTY)


def test_ablation_guard_records_that_a_no_regression_bar_is_free() -> None:
    """A `>= 0` bar is cleared by a zero uplift, and the record says so.

    Not a rejection — the gate is a regression tripwire and still
    catches one — but a reader must not read its green as positive
    evidence about the feature.
    """
    rec = _Recorder()
    arms = AblationArms(
        shipped=0.0, ablated=0.0, without_row_scores=[1.0] * 30
    )
    guard_ablation_gate(
        module="synthetic_ablation",
        rows=[{"id": f"r{i}"} for i in range(30)],
        arms=lambda: arms,
        bar=bar_at_least(0.0),
        record_property=rec,
    )
    assert "null_clears_bar=True" in rec.value(BENCH_MEASUREMENT_PROPERTY)
    assert "verdict=ACCEPT" in rec.value(BENCH_MEASUREMENT_PROPERTY)


def test_a_prefilter_rejection_does_not_run_the_shipped_arm() -> None:
    """Milliseconds before a store-per-row rebuild, and it reports as such."""
    ran = []
    rec = _Recorder()

    def shipped() -> float:
        ran.append(1)
        return 0.99

    with pytest.raises(pytest.fail.Exception, match="separability"):
        guard_ranking_gate(
            module="synthetic_ranking",
            rows=_ranking_rows(30, pool=10, gold=10, k=10),
            shipped=shipped,
            bar=bar_at_least(0.5),
            metric=precision_at_k,
            record_property=rec,
            gold_key="expected_top_k",
            pool_key="beliefs",
        )
    assert ran == []
    assert "shipped=unmeasured" in rec.value(BENCH_MEASUREMENT_PROPERTY)
