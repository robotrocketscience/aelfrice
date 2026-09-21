"""A bench-gate corpus counts only when it defeats its own null model (#1581).

Three separate things are pinned here, because three separate ways of
losing the guard were observed:

* **Declaration.** Every module under `tests/bench_gate/` names a family
  and a null model in `GATE_DECLARATIONS`. A new gate that declares
  neither fails here rather than shipping as an unguarded "executed"
  count.
* **Wiring.** A declaration is not the guard. Every non-exempt gate
  module must actually call one, in the gate, on the same rows — which
  is checked by parsing the file rather than by trusting the registry.
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

import ast
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
    pool_shape,
    precision_at_k,
    structural_prefilter,
)
from tests.conftest import (
    BENCH_MEASUREMENT_PROPERTY,
    BENCH_NULL_VERDICT_PROPERTY,
    BENCH_VERDICT_ACCEPT,
    BENCH_VERDICT_REJECT,
    BENCH_VERDICT_UNVERIFIED,
    NO_VERDICT_RECORDED,
    exempt_gate_modules,
    tally_bench_reports,
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
# Wiring
# ---------------------------------------------------------------------------


def _calls_a_guard(path: Path) -> bool:
    """True when the module calls one of the family guards.

    Parsed rather than grepped: `guard_ranking_gate` appears in this
    file's own prose and in the registry's, and a substring match would
    read a docstring as a wired gate.
    """
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name.startswith("guard_") and name.endswith("_gate"):
            return True
    return False


def test_every_scored_gate_actually_calls_its_guard() -> None:
    """AC2. The declaration is not the guard; the call is.

    A registry entry with no call site is the failure mode this closes:
    the tier would report a declared family for a gate that never ran a
    null model, which is worse than an undeclared gate because it reads
    as covered.
    """
    missing = [
        path.name
        for path in _gate_files()
        if GATE_DECLARATIONS[path.stem].family is not Family.EXEMPT
        and not _calls_a_guard(path)
    ]
    assert not missing, (
        f"bench-gate modules that declare a null model but never run one: "
        f"{missing}. The null-model run happens in the gate, on the same "
        f"rows, every time the gate runs."
    )


def _loaded_corpus_modules(path: Path) -> set[str]:
    """Every literal module name the file passes to `load_corpus_module`."""
    loaded: set[str] = set()
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "load_corpus_module" or len(node.args) < 2:
            continue
        arg = node.args[1]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            loaded.add(arg.value)
    return loaded


def test_every_declaration_names_the_corpus_its_gate_actually_loads() -> None:
    """The registry is advertised as the single source of truth.

    Nothing consumed `corpus_module` before this, so a declaration could
    name one corpus while the gate graded another and the tier printed
    both names without noticing. The drift is invisible by construction
    unless the declaration is compared with the call.
    """
    drifted = {
        path.name: (decl.corpus_module, sorted(loaded))
        for path in _gate_files()
        for decl in [GATE_DECLARATIONS[path.stem]]
        if decl.family is not Family.EXEMPT
        for loaded in [_loaded_corpus_modules(path)]
        if loaded != {decl.corpus_module}
    }
    assert not drifted, (
        f"gate modules whose declared corpus_module is not the one they "
        f"load: {drifted}. Each entry reads "
        f"file: (declared, actually loaded)."
    )


def test_no_bench_gate_writes_its_own_corpus_skip() -> None:
    """Corpus-state skips go through `skip_corpus_module` (AC5).

    A bespoke `pytest.skip` is how an under-`MIN_ROWS` module went
    unreported: the summary parses one reason shape, and a message that
    does not match it prints neither "executed" nor "no verdict".
    """
    offenders: list[str] = []
    for path in _gate_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "skip"):
                continue
            text = ast.unparse(node)
            if "corpus" in text:
                offenders.append(f"{path.name}: {text.splitlines()[0][:90]}")
    assert not offenders, (
        "corpus-state skips must use tests.conftest.skip_corpus_module or "
        "require_min_rows so the tier summary can classify them:\n  "
        + "\n  ".join(offenders)
    )


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


# ---------------------------------------------------------------------------
# The verdict survives the shipped arm
# ---------------------------------------------------------------------------


class _ShippedArmFailure(RuntimeError):
    """Stands in for the import and attribute errors gates hit for real."""


@pytest.mark.parametrize(
    "guard",
    ["ranking", "classification", "ablation"],
)
def test_a_shipped_arm_that_raises_still_records_a_verdict(guard: str) -> None:
    """AC2. The null model ran, so its verdict is evidence either way.

    Before this the record was written after the shipped arm, so any
    exception in it discarded the verdict — and a bench-gated report
    with no verdict was counted as executed.
    """
    rec = _Recorder()

    def boom() -> float:
        raise _ShippedArmFailure("the graded code does not exist")

    with pytest.raises(_ShippedArmFailure):
        if guard == "ranking":
            guard_ranking_gate(
                module="synthetic",
                rows=_ranking_rows(30, pool=20, gold=3, k=5),
                shipped=boom,
                bar=bar_at_least(0.5),
                metric=precision_at_k,
                record_property=rec,
                gold_key="expected_top_k",
                pool_key="beliefs",
            )
        elif guard == "classification":
            guard_classification_gate(
                module="synthetic",
                rows=_labelled({"positive": 11, "negative": 10, "neutral": 9}),
                shipped=boom,
                bar=bar_at_least(0.5),
                score_constant=lambda label: 0.41,
                record_property=rec,
            )
        else:
            guard_ablation_gate(
                module="synthetic",
                rows=[{"id": f"r{i}"} for i in range(30)],
                arms=boom,
                bar=bar_at_least(0.05),
                record_property=rec,
            )

    verdict = rec.value(BENCH_NULL_VERDICT_PROPERTY)
    assert BENCH_VERDICT_UNVERIFIED in verdict
    assert "_ShippedArmFailure" in verdict
    assert "errored(_ShippedArmFailure)" in rec.value(BENCH_MEASUREMENT_PROPERTY)


def test_a_rejected_corpus_never_runs_the_shipped_arm() -> None:
    """The verdict is about the corpus, so it is settled first."""
    ran: list[int] = []
    rec = _Recorder()

    def shipped() -> float:
        ran.append(1)
        return 0.93

    rows = _labelled({"negative": 27, "positive": 3})
    with pytest.raises(pytest.fail.Exception, match="constant predictor"):
        guard_classification_gate(
            module="synthetic",
            rows=rows,
            shipped=shipped,
            bar=bar_at_least(0.5),
            score_constant=lambda label: majority_constant_accuracy(rows)[0],
            record_property=rec,
        )
    assert ran == []
    assert "shipped=unmeasured" in rec.value(BENCH_MEASUREMENT_PROPERTY)


def test_ablation_guard_refuses_an_unchecked_no_pool_claim() -> None:
    """`gold_key=None` on an ablation gate is a claim, not a switch.

    The family where it matters most: "gold == pool" is what produces
    the +1.000 tautology the `without_rate` floor exists to reject, so a
    pool-shaped corpus must wire the keys and run the pre-filters.
    """
    rows = _ranking_rows(30, pool=10, gold=10, k=10)
    with pytest.raises(AssertionError, match="declared no candidate pool"):
        guard_ablation_gate(
            module="synthetic",
            rows=rows,
            arms=lambda: AblationArms(
                shipped=1.0, ablated=0.0, without_row_scores=[1.0] * 30
            ),
            bar=bar_at_least(0.05),
            record_property=_Recorder(),
        )


def test_a_belief_pool_alone_is_not_a_candidate_pool() -> None:
    """The `compression_a4_fidelity` shape, which must still pass.

    Its rows seed a store from `beliefs` and score free-text answers
    against it, so the gold is not drawn from the pool and
    `len(gold) < len(pool)` would compare two different populations.
    Keying the check on key presence rejected this corpus; keying it on
    "the gold entries are pool ids" does not.
    """
    rows = _ranking_rows(30, pool=6, gold=0, k=3)
    for row in rows:
        row.pop("expected_top_k")
        row["expected_answers"] = ["a binary heap gives log-time extract-min"]
    assert pool_shape(rows) is None
    rec = _Recorder()
    guard_ablation_gate(
        module="synthetic",
        rows=rows,
        arms=lambda: AblationArms(
            shipped=0.61, ablated=0.48, without_row_scores=[1.0] * 30
        ),
        bar=bar_at_least(0.05),
        record_property=rec,
    )
    assert "ACCEPT" in rec.value(BENCH_NULL_VERDICT_PROPERTY)


def test_ablation_guard_runs_the_prefilters_once_the_keys_are_wired() -> None:
    """And the wired path still rejects the degenerate shape."""
    rec = _Recorder()
    with pytest.raises(pytest.fail.Exception, match="separability"):
        guard_ablation_gate(
            module="synthetic",
            rows=_ranking_rows(30, pool=10, gold=10, k=10),
            arms=lambda: AblationArms(
                shipped=1.0, ablated=0.0, without_row_scores=[1.0] * 30
            ),
            bar=bar_at_least(0.05),
            record_property=rec,
            gold_key="expected_top_k",
            pool_key="beliefs",
        )
    assert "REJECT" in rec.value(BENCH_NULL_VERDICT_PROPERTY)


# ---------------------------------------------------------------------------
# What the tier counts
# ---------------------------------------------------------------------------


class _Report:
    """The two attributes `tally_bench_reports` reads off a pytest report."""

    def __init__(self, nodeid: str, properties: list[tuple[str, str]]) -> None:
        self.nodeid = nodeid
        self.user_properties = properties


def _verdict_property(module: str, state: str, why: str = "") -> tuple[str, str]:
    return (BENCH_NULL_VERDICT_PROPERTY, f"{module}|{state}|{why}")


def test_a_bench_gated_report_with_no_verdict_is_not_executed() -> None:
    """AC3. The hole the AST wiring check cannot see.

    `_calls_a_guard` returns True for a call site that is present in the
    file; it cannot tell a call that runs from one behind `if False:`.
    A guard that does not run leaves no verdict, so requiring the
    property at summary time is what stops the unrun gate being counted
    as evidence.
    """
    node = "tests/bench_gate/test_sentiment.py::test_x"
    tally = tally_bench_reports([_Report(node, [])], exempt_modules=frozenset())
    assert tally.executed == 0
    assert tally.unverified == {node: NO_VERDICT_RECORDED}
    assert tally.rejected == {}


def test_an_exempt_module_needs_no_verdict_to_count() -> None:
    """An exemption is declared and reprinted; it is not an unrun guard."""
    tally = tally_bench_reports(
        [_Report("tests/bench_gate/test_hrr_cold_start.py::test_x", [])],
        exempt_modules=frozenset({"test_hrr_cold_start"}),
    )
    assert tally.executed == 1
    assert tally.unverified == {}


def test_an_accept_verdict_is_the_only_state_counted_as_executed() -> None:
    tally = tally_bench_reports(
        [
            _Report(
                "tests/bench_gate/test_sentiment.py::test_x",
                [_verdict_property("sentiment", BENCH_VERDICT_ACCEPT)],
            )
        ],
        exempt_modules=frozenset(),
    )
    assert tally.executed == 1
    assert tally.unverified == {}
    assert tally.rejected == {}


def test_a_reject_verdict_is_neither_executed_nor_unverified() -> None:
    tally = tally_bench_reports(
        [
            _Report(
                "tests/bench_gate/test_query_strategy.py::test_x",
                [
                    _verdict_property(
                        "query_strategy", BENCH_VERDICT_REJECT, "skewed"
                    )
                ],
            )
        ],
        exempt_modules=frozenset(),
    )
    assert tally.executed == 0
    assert tally.rejected == {"query_strategy": "skewed"}
    assert tally.unverified == {}


def test_an_unverified_verdict_records_the_null_run_without_counting() -> None:
    """A shipped arm that raised graded nothing, but its null model ran."""
    tally = tally_bench_reports(
        [
            _Report(
                "tests/bench_gate/test_sentiment.py::test_x",
                [
                    _verdict_property(
                        "sentiment", BENCH_VERDICT_UNVERIFIED, "shipped arm raised"
                    )
                ],
            )
        ],
        exempt_modules=frozenset(),
    )
    assert tally.executed == 0
    assert tally.unverified == {"sentiment": "shipped arm raised"}
    assert tally.rejected == {}


def test_the_tally_keeps_every_measurement_line() -> None:
    tally = tally_bench_reports(
        [
            _Report(
                "tests/bench_gate/test_sentiment.py::test_x",
                [
                    (BENCH_MEASUREMENT_PROPERTY, "sentiment: one"),
                    (BENCH_MEASUREMENT_PROPERTY, "sentiment: two"),
                    _verdict_property("sentiment", BENCH_VERDICT_ACCEPT),
                ],
            )
        ],
        exempt_modules=frozenset(),
    )
    assert tally.measurements == ["sentiment: one", "sentiment: two"]


def test_exempt_gate_modules_matches_the_registry() -> None:
    """The summary's exemption list is the registry's, not a second copy."""
    assert exempt_gate_modules() == frozenset(
        stem
        for stem, decl in GATE_DECLARATIONS.items()
        if decl.family is Family.EXEMPT
    )
    assert exempt_gate_modules(), "the registry declares at least one exemption"
