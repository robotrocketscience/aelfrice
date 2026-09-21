"""Null-model preconditions for the bench-gate tier (#1581).

A bench gate reports "N tests executed against the corpus". That number
is evidence only if the corpus can separate the shipped implementation
from a model that *cannot represent the distinction the gate measures*.
Operator ruling 2026-09-20 makes the check a precondition to counting
rather than an advisory:

    A corpus counts only when the gate's own metric, recomputed with the
    shipped implementation replaced by the module's declared null model,
    fails the gate's bar — while the shipped implementation passes it.
    The null-model run executes in the same test, on the same rows,
    every time the gate runs.

`tests/bench_gate/test_directive_detection.py::test_directive_corpus_defeats_a_first_token_baseline`
is the in-repo precedent (#374, #1349). This module generalises it: the
same "a degenerate model must not clear the bar" shape, declared once
per gate module, run inside the gate rather than beside it, and
recorded on the green path as well as the red one.

Three things live here.

**The declarations.** `GATE_DECLARATIONS` names, for every module under
`tests/bench_gate/`, which family it belongs to and what its null model
is. `tests/test_bench_gate_null_model_1581.py` fails when a module has
no entry, so a new gate cannot omit one.

**The structural pre-filters.** Two cheap shape checks that run before
any null model, because they cost milliseconds and a corpus that fails
them cannot be rescued by a better metric. They are reported by name, so
a rejection says which one fired.

**The guards.** One per family. Each computes the null score on the same
rows the gate scored, records the shipped score and the null score
through `record_property`, and fails the test when the corpus does not
count.

Why the guards assert only the null half of the rule: the shipped half
is each gate's own existing assertion, and two of them are deliberately
inverted tripwires (#1502) whose shipped arm is *expected* not to clear.
Asserting "shipped passes" here would turn those ratified tripwires red
on the state they were inverted to report.
"""
from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

import pytest

from tests.bench_protocol import (
    BENCH_MEASUREMENT_PROPERTY,
    BENCH_NULL_VERDICT_PROPERTY,
    BENCH_VERDICT_ACCEPT,
    BENCH_VERDICT_REJECT,
    BENCH_VERDICT_UNVERIFIED,
)

Row = Mapping[str, object]
RecordProperty = Callable[[str, object], None]
_T = TypeVar("_T")


class Family(str, Enum):
    """What kind of claim a gate makes, which fixes its null model."""

    RANKING = "ranking"
    """Scores an ordering of a per-row candidate pool against gold."""

    CLASSIFICATION = "classification"
    """Scores a per-row label prediction against a gold label."""

    ABLATION = "ablation"
    """Scores the delta between a treatment arm and an ablated arm."""

    EXEMPT = "exempt"
    """No null model is constructible from these rows; says why.

    An exemption is a declaration, not an omission: it names the reason
    in the registry and the registry test reprints them, so an exempt
    gate is visible rather than silently counted as validated.
    """


@dataclass(frozen=True)
class GateDeclaration:
    """One gate module's family and null model, declared in one place."""

    family: Family
    null_model: str
    corpus_module: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.family is Family.EXEMPT:
            if not self.reason:
                raise ValueError("an EXEMPT declaration must give a reason")
        elif not self.corpus_module:
            raise ValueError(
                f"a {self.family.value} declaration must name its corpus module"
            )
        if not self.null_model:
            raise ValueError("every declaration must state its null model")


# Keyed by the test module's file stem. Exhaustive over
# `tests/bench_gate/test_*.py` — `test_bench_gate_null_model_1581.py`
# fails when a file has no entry here, which is what stops a new gate
# shipping without a declared null model.
GATE_DECLARATIONS: dict[str, GateDeclaration] = {
    # -- ranking ----------------------------------------------------------
    "test_query_strategy": GateDeclaration(
        family=Family.RANKING,
        corpus_module="query_strategy",
        null_model=(
            "the row's candidate pool in deterministic shuffled order, scored "
            "with the gate's own NDCG@k against the gate's `> 0.0` floor"
        ),
    ),
    "test_wonder_online": GateDeclaration(
        family=Family.RANKING,
        corpus_module="wonder_online",
        null_model=(
            "the row's belief pool in deterministic shuffled order, taking the "
            "same top-k and scoring the same ≥1-expected-candidate row hit"
        ),
    ),
    # -- classification ---------------------------------------------------
    "test_directive_detection": GateDeclaration(
        family=Family.CLASSIFICATION,
        corpus_module="directive_detection",
        null_model=(
            "a constant predictor returning the corpus's majority label; the "
            "stronger first-token baseline over K=200 salted partitions is "
            "enforced alongside it in the same module (#374, #1349)"
        ),
    ),
    "test_sentiment": GateDeclaration(
        family=Family.CLASSIFICATION,
        corpus_module="sentiment",
        null_model="a constant predictor returning the corpus's majority label",
    ),
    "test_contradiction": GateDeclaration(
        family=Family.CLASSIFICATION,
        corpus_module="contradiction",
        null_model="a constant predictor returning the corpus's majority label",
    ),
    "test_contradiction_v3": GateDeclaration(
        family=Family.CLASSIFICATION,
        corpus_module="contradiction",
        null_model=(
            "a constant predictor returning the corpus's majority label, scored "
            "as the min headroom against the precision and recall floors"
        ),
    ),
    # -- ablation / uplift ------------------------------------------------
    "test_bfs_multihop_derived_from": GateDeclaration(
        family=Family.ABLATION,
        corpus_module="derived_from_edge",
        null_model="the ablated arm (DERIVED_FROM edge weight zeroed)",
    ),
    "test_bfs_multihop_implements": GateDeclaration(
        family=Family.ABLATION,
        corpus_module="implements_edge",
        null_model="the ablated arm (IMPLEMENTS edge weight zeroed)",
    ),
    "test_bfs_multihop_relates_to": GateDeclaration(
        family=Family.ABLATION,
        corpus_module="bfs_relates_to",
        null_model="the ablated arm (RELATES_TO edge weight zeroed)",
    ),
    "test_bfs_multihop_temporal_next": GateDeclaration(
        family=Family.ABLATION,
        corpus_module="temporal_next_edge",
        null_model="the ablated arm (TEMPORAL_NEXT edge weight zeroed)",
    ),
    "test_bfs_multihop_tests": GateDeclaration(
        family=Family.ABLATION,
        corpus_module="tests_edge",
        null_model="the ablated arm (TESTS edge weight zeroed)",
    ),
    "test_reason": GateDeclaration(
        family=Family.ABLATION,
        corpus_module="reasoning",
        null_model="the ablated arm (chain expansion disabled)",
    ),
    "test_retrieve_uplift": GateDeclaration(
        family=Family.ABLATION,
        corpus_module="retrieve_uplift",
        null_model="the ablated arm (all v1.7 flags off — the baseline arm)",
    ),
    "test_compression_a2_recall": GateDeclaration(
        family=Family.ABLATION,
        corpus_module="compression_a2_recall",
        null_model="the ablated arm (use_type_aware_compression off)",
    ),
    "test_compression_a4_fidelity": GateDeclaration(
        family=Family.ABLATION,
        corpus_module="compression_a4_fidelity",
        null_model="the ablated arm (use_type_aware_compression off)",
    ),
    "test_doc_linker": GateDeclaration(
        family=Family.ABLATION,
        corpus_module="doc_linker",
        null_model="the ablated arm (doc linking off)",
    ),
    "test_intentional_clustering": GateDeclaration(
        family=Family.ABLATION,
        corpus_module="multi_fact",
        null_model="the ablated arm (intentional clustering off)",
    ),
    "test_edge_rerank_potentially_stale": GateDeclaration(
        family=Family.ABLATION,
        corpus_module="bfs_potentially_stale",
        null_model="the ablated arm (POTENTIALLY_STALE rerank off)",
    ),
    # -- exempt -----------------------------------------------------------
    "test_commit_intent": GateDeclaration(
        family=Family.EXEMPT,
        null_model="none",
        reason=(
            "in-file parametrised unit tests over hand-written messages; no "
            "corpus rows and no marker, so nothing in the tier counts it"
        ),
    ),
    "test_compression_uplift": GateDeclaration(
        family=Family.EXEMPT,
        null_model="none",
        reason=(
            "the metric is a within-row token-count ratio against no labelled "
            "gold, and the ablated arm is the ratio's own denominator — every "
            "null constructible from these rows scores exactly 0.0 reduction "
            "by definition, so the check would report a pass it did not "
            "perform (#1160). Its corpus is mounted and unguarded"
        ),
    ),
    "test_cluster_edge_floor": GateDeclaration(
        family=Family.EXEMPT,
        null_model="none",
        reason=(
            "asserts monotonicity of the clusterer across two floors, not a "
            "score against a bar; a shuffled or constant model has no verdict "
            "to compare"
        ),
    ),
    "test_entity_persist_g2_mixed_corpus": GateDeclaration(
        family=Family.EXEMPT,
        null_model="none",
        reason=(
            "drives a synthetic in-repo store built by "
            "`_entity_persist_mixed_store.py`, not labelled corpus rows; the "
            "population is generated by the test, so a null model would be "
            "scored against the generator rather than against a corpus"
        ),
    ),
    "test_hrr_cold_start": GateDeclaration(
        family=Family.EXEMPT,
        null_model="none",
        reason="wall-clock latency ceilings on a synthetic store; no labelled rows",
    ),
    "test_rerank_relevance": GateDeclaration(
        family=Family.EXEMPT,
        null_model="none",
        reason=(
            "row-shape smoke harness with no metric and no bar; the four "
            "consumers that grade these rows carry their own declarations"
        ),
    ),
    "test_wonder_consolidation": GateDeclaration(
        family=Family.EXEMPT,
        null_model="none",
        reason=(
            "asserts only that the strategy returns a number for every row — "
            "#228 never defined the metric, so there is no bar to defeat"
        ),
    ),
}


def exempt_gate_modules() -> frozenset[str]:
    """File stems of the bench-gate modules declared `Family.EXEMPT`.

    Here rather than in `tests/conftest.py`, which is the caller: the
    tier summary asks which gates have no guard to run, and the answer
    is a query over this registry. Reading it from the summary meant
    `conftest` importing this module while this module imported
    `conftest` for the verdict strings — a cycle, held together by the
    import being deferred into the function body. The strings moved to
    `tests/bench_protocol.py` and the query moved to the registry, so
    the dependency now runs one way: conftest -> null_model ->
    bench_protocol.
    """
    return frozenset(
        stem
        for stem, decl in GATE_DECLARATIONS.items()
        if decl.family is Family.EXEMPT
    )


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bar:
    """A gate's bar, as a predicate over a scalar score plus its prose."""

    text: str
    clears: Callable[[float], bool]


def bar_above(threshold: float) -> Bar:
    return Bar(f"> {threshold:g}", lambda s: s > threshold)


def bar_at_least(threshold: float) -> Bar:
    return Bar(f">= {threshold:g}", lambda s: s >= threshold)


def bar_headroom(text: str) -> Bar:
    """A conjunction of floors, reduced to `min(score - floor) >= 0`.

    Two gates state their bar as "precision >= P and recall >= R". A
    scalar headroom keeps one comparison shape for every family while
    preserving the conjunction: it is non-negative exactly when both
    floors are met, so the null clears the bar exactly when it would
    have cleared both.
    """
    return Bar(text, lambda s: s >= 0.0)


# ---------------------------------------------------------------------------
# Structural pre-filters
# ---------------------------------------------------------------------------

SEPARABILITY_MIN_SHARE = 0.90
"""Share of rows that must satisfy `len(gold) < len(pool)` and `k < len(pool)`.

Below it the corpus cannot lose a distractor: with the gold set equal to
the whole pool, any ordering scores near the ceiling. Measured on
`query_strategy` 2026-09-20: 0% and 7% respectively, against a gate whose
floor a deterministic shuffle clears by a wide margin.
"""

GOLD_SKEW_FACTOR = 3.0
"""Cap on `max(len(gold)) / median(len(gold))`.

Micro-averaged metrics weight a row by its target count, so one 60-target
row in a 30-row corpus supplies most of the denominator and the corpus
becomes, in effect, one row. A `MIN_ROWS` floor counts rows and does not
defend against that.
"""


@dataclass(frozen=True)
class PrefilterRejection:
    """Which pre-filter rejected a corpus, and on what numbers."""

    name: str
    detail: str


def structural_prefilter(
    rows: Sequence[Row],
    *,
    gold_key: str,
    pool_key: str,
    k_key: str = "k",
    default_k: int = 10,
) -> PrefilterRejection | None:
    """Run the two cheap shape checks. Return the first rejection, or None.

    Ordered cheapest-first and run before any null model, because both
    cost milliseconds against a null model that rebuilds a store per row.

    Raises when a row lacks the declared keys: a pre-filter that silently
    passes on rows it could not read is the defect it exists to catch.
    """
    if not rows:
        raise ValueError("structural_prefilter needs at least one row")

    gold_sizes: list[int] = []
    separable = 0
    for row in rows:
        if gold_key not in row or pool_key not in row:
            raise KeyError(
                f"row {row.get('id', '<no id>')!r} lacks {gold_key!r} or "
                f"{pool_key!r}; the pre-filter cannot read this corpus"
            )
        gold = row[gold_key]
        pool = row[pool_key]
        assert isinstance(gold, list) and isinstance(pool, list)
        k = int(row.get(k_key, default_k))  # type: ignore[arg-type]
        gold_sizes.append(len(gold))
        if len(gold) < len(pool) and k < len(pool):
            separable += 1

    share = separable / len(rows)
    if share < SEPARABILITY_MIN_SHARE:
        return PrefilterRejection(
            name="separability",
            detail=(
                f"only {share:.1%} of {len(rows)} rows satisfy "
                f"len(gold) < len(pool) and k < len(pool) "
                f"(floor {SEPARABILITY_MIN_SHARE:.0%}); the gold set is the "
                f"candidate pool, so no distractor can be lost and any "
                f"ordering scores near the ceiling"
            ),
        )

    median_gold = statistics.median(gold_sizes)
    max_gold = max(gold_sizes)
    if median_gold > 0 and max_gold > GOLD_SKEW_FACTOR * median_gold:
        return PrefilterRejection(
            name="gold-skew",
            detail=(
                f"max(len(gold))={max_gold} exceeds "
                f"{GOLD_SKEW_FACTOR:g}x median(len(gold))={median_gold:g}; "
                f"a micro-averaged metric lets the largest row dominate the "
                f"denominator, which a MIN_ROWS floor does not defend against"
            ),
        )
    return None


# ---------------------------------------------------------------------------
# Null models
# ---------------------------------------------------------------------------

SHUFFLE_SALT = "aelfrice-null-model-1581"
"""Salt for the ranking null model's per-row seed.

Seeded on the row id so the shuffle is stable across runs, independent
of file order, and independent of how many rows were loaded — a null
score that moved between runs would be unusable as a precondition.
"""


def shuffled_pool(row: Row, *, pool_key: str, id_key: str = "id") -> list[str]:
    """The row's candidate pool in deterministic shuffled order."""
    pool = row[pool_key]
    assert isinstance(pool, list)
    ids = [str(b[id_key]) for b in pool]
    random.Random(f"{SHUFFLE_SALT}:{row.get('id', '')}").shuffle(ids)
    return ids


def shuffled_ranker_score(
    rows: Sequence[Row],
    *,
    metric: Callable[[list[str], list[str], int], float],
    gold_key: str,
    pool_key: str,
    k_key: str = "k",
    default_k: int = 10,
) -> float:
    """Mean of the gate's own metric over shuffled candidate pools."""
    total = 0.0
    for row in rows:
        gold = row[gold_key]
        assert isinstance(gold, list)
        k = int(row.get(k_key, default_k))  # type: ignore[arg-type]
        total += metric(shuffled_pool(row, pool_key=pool_key), list(gold), k)
    return total / len(rows) if rows else 0.0


def precision_at_k(ranked: list[str], gold: list[str], k: int) -> float:
    """Share of the top-k that is gold. Undefined k<=0 scores 0.0."""
    if k <= 0:
        return 0.0
    gold_set = set(gold)
    return sum(1 for bid in ranked[:k] if bid in gold_set) / k


def majority_label(rows: Sequence[Row], *, label_key: str = "label") -> str:
    """The corpus's most frequent label, ties broken by sort order."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row[label_key])] = counts.get(str(row[label_key]), 0) + 1
    return max(sorted(counts), key=lambda lab: counts[lab])


def constant_predictor_accuracy(
    rows: Sequence[Row], label: str, *, label_key: str = "label"
) -> float:
    """Accuracy of answering `label` on every row."""
    return sum(1 for row in rows if str(row[label_key]) == label) / len(rows)


def majority_constant_accuracy(
    rows: Sequence[Row], *, label_key: str = "label"
) -> tuple[float, str]:
    """Accuracy of the constant majority-label predictor, and that label."""
    label = majority_label(rows, label_key=label_key)
    return constant_predictor_accuracy(rows, label, label_key=label_key), label


MIN_WITHOUT_SHARE = 0.10
"""Share of gradeable rows the ablated arm must score above zero on.

The rejection this implements: "the feature is the only path to the
gold". A corpus built so that nothing reaches a target without the
treatment yields uplift +1.000 on every run, which grades the weight
table's own arithmetic rather than the feature. One or two accidental
hits do not clear that — at the tier's 30-row floor this asks for three
rows — while any corpus with a real distractor population clears it
easily.

Declared, not derived: no measurement fixes this number, and raising it
is a corpus-design decision rather than a statistical one.
"""


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

# Re-exported under the short names the guards read with. The strings
# live in `tests.bench_protocol` — a leaf both this module and the tier
# summary import — because the summary classifies on them and must not
# read a second copy that can drift from the one the guards write.
ACCEPT = BENCH_VERDICT_ACCEPT
REJECT = BENCH_VERDICT_REJECT
UNVERIFIED = BENCH_VERDICT_UNVERIFIED

_POOL_KEYS = ("beliefs",)
"""Row keys that hold a candidate pool: a list of objects carrying an id."""


@dataclass(frozen=True)
class RankingResult:
    """What a ranking gate's guard measured."""

    shipped: float
    null: float


@dataclass(frozen=True)
class AblationArms:
    """Both arms of an ablation gate, as the gate's own driver produced them.

    `without_row_scores` is the ablated arm *per row*, not the mean: the
    rejection this family implements asks on what share of rows the
    feature was not the only path to the gold, and a mean cannot answer
    that.
    """

    shipped: float
    ablated: float
    without_row_scores: Sequence[float]

    @property
    def uplift(self) -> float:
        """The quantity these gates state their bar in."""
        return self.shipped - self.ablated


def pool_shape(rows: Sequence[Row]) -> tuple[str, str] | None:
    """The `(pool_key, gold_key)` pair whose gold is drawn from the pool.

    Detected rather than enumerated. Carrying a `beliefs` list does not
    make a corpus ranking-shaped: `compression_a4_fidelity` rows seed a
    store from `beliefs` and score free-text answers against it, so
    `len(gold) < len(pool)` there compares two different populations.
    What makes the structural pre-filters apply is that the gold *is* a
    subset of the pool's ids, so that is the condition tested — over
    every list-of-strings key, with no gold-key list to keep in sync.
    """
    for pool_key in _POOL_KEYS:
        ids = {
            str(item["id"])
            for row in rows
            if isinstance(row.get(pool_key), list)
            for item in row[pool_key]  # type: ignore[index]
            if isinstance(item, Mapping) and "id" in item
        }
        if not ids:
            continue
        for gold_key in sorted({key for row in rows for key in row}):
            if gold_key == pool_key:
                continue
            for row in rows:
                gold = row.get(gold_key)
                if isinstance(gold, list) and any(
                    isinstance(g, str) and g in ids for g in gold
                ):
                    return pool_key, gold_key
    return None


def _assert_no_pool_shape(rows: Sequence[Row], module: str) -> None:
    """Fail when a gate declares no pool while its rows carry one.

    `gold_key=None` turns the structural pre-filters off, so it has to
    be a claim about the corpus rather than a switch. A ranking-shaped
    corpus that declared itself pool-free would skip both filters
    silently, which is the same invisibility #1581 closes elsewhere.

    Called from the classification guard, which never runs the filters,
    and from the ablation guard when it was given no keys — the two
    paths that can reach a graded corpus without a shape check.
    """
    shape = pool_shape(rows)
    if shape:
        found_pool, found_gold = shape
        raise AssertionError(
            f"gate {module!r} declared no candidate pool, but its "
            f"{found_gold!r} entries are ids drawn from its {found_pool!r} "
            f"pool — pass gold_key={found_gold!r} and "
            f"pool_key={found_pool!r} so the structural pre-filters run, "
            f"with k_key naming the gate's own cutoff"
        )


SHIPPED_ARM_RAISED = (
    "the null model ran, but the shipped arm raised before it scored: "
)
"""Why an UNVERIFIED verdict was recorded instead of ACCEPT."""


def _measure(arm: Callable[[], _T]) -> tuple[_T | None, BaseException | None]:
    """Run a gate's shipped arm, capturing whatever it raises.

    The caller re-raises through `_verdict`, which is the only place a
    verdict is recorded. Letting the exception out of the guard directly
    is the defect #1581 closes: the record is written after the arm, so
    an arm that raised discarded the null-model verdict entirely and the
    tier counted the verdict-less test as executed.
    """
    try:
        return arm(), None
    except BaseException as exc:  # noqa: BLE001 - re-raised below, unchanged
        return None, exc


def _verdict(
    *,
    module: str,
    family: Family,
    shipped_score: float | None,
    null_score: float,
    bar: Bar,
    rejection: str | None,
    extra: str,
    record_property: RecordProperty,
    shipped_error: BaseException | None = None,
) -> None:
    """Record both scores, then fail when the corpus does not count.

    Recorded before the assertion so a green run carries the evidence
    too: a gate whose numbers exist only inside a failure message leaves
    a release reviewer unable to tell a degenerate +1.000 uplift from a
    real +0.06.

    `shipped_score` is None when the shipped arm did not run or did not
    return. It did not run when a structural pre-filter or the null
    model already rejected the corpus — the filters cost milliseconds
    and the arms rebuild a store per row, so the order is deliberate and
    the record says "unmeasured" rather than inventing a number. It did
    not return when `shipped_error` is set, which is UNVERIFIED: the
    null model is still evidence, and the gate still fails, but nothing
    was graded so nothing is counted.
    """
    if rejection:
        state = REJECT
    elif shipped_error is not None:
        state = UNVERIFIED
    else:
        state = ACCEPT
    if shipped_score is not None:
        shipped_text = f"{shipped_score:.4f}"
    elif shipped_error is not None:
        shipped_text = f"errored({type(shipped_error).__name__})"
    else:
        shipped_text = "unmeasured"
    line = (
        f"{module}: family={family.value} shipped={shipped_text} "
        f"null={null_score:.4f} bar={bar.text} "
        f"null_clears_bar={bar.clears(null_score)} verdict={state}"
    )
    if extra:
        line = f"{line} {extra}"
    why = rejection or ""
    if shipped_error is not None:
        why = f"{SHIPPED_ARM_RAISED}{type(shipped_error).__name__}: {shipped_error}"
    record_property(BENCH_MEASUREMENT_PROPERTY, line)
    record_property(BENCH_NULL_VERDICT_PROPERTY, f"{module}|{state}|{why}")
    if rejection:
        pytest.fail(
            f"corpus {module!r} does not count: {rejection}\n"
            f"  {line}\n"
            f"  Per the #1581 ruling the null-model check is a precondition "
            f"to counting, not an advisory. This gate has no verdict until "
            f"the corpus is rebuilt; do not read its other assertions as "
            f"evidence, and do not weaken this one."
        )
    if shipped_error is not None:
        raise shipped_error


def guard_ranking_gate(
    *,
    module: str,
    rows: Sequence[Row],
    shipped: Callable[[], float],
    bar: Bar,
    metric: Callable[[list[str], list[str], int], float],
    record_property: RecordProperty,
    gold_key: str,
    pool_key: str,
    k_key: str = "k",
    default_k: int = 10,
    extra: str = "",
) -> RankingResult:
    """Ranking family: a deterministic shuffle of the row's candidate pool.

    `shipped` is a thunk rather than a number so the structural
    pre-filters can reject before it runs. They cost milliseconds; the
    shipped arm drives `retrieve()` once per row.
    """
    pre = structural_prefilter(
        rows, gold_key=gold_key, pool_key=pool_key, k_key=k_key, default_k=default_k
    )
    # Scored before the pre-filter verdict is acted on: shuffling a
    # per-row id list costs microseconds, so a rejected corpus still
    # gets its null score onto the record, and the reviewer can see how
    # far past the bar the degenerate model went.
    null_score = shuffled_ranker_score(
        rows,
        metric=metric,
        gold_key=gold_key,
        pool_key=pool_key,
        k_key=k_key,
        default_k=default_k,
    )
    if pre:
        _verdict(
            module=module,
            family=Family.RANKING,
            shipped_score=None,
            null_score=null_score,
            bar=bar,
            rejection=(
                f"structural pre-filter {pre.name!r} rejected it — {pre.detail}"
            ),
            extra=f"n_rows={len(rows)} prefilter={pre.name} {extra}".strip(),
            record_property=record_property,
        )
    rejection = None
    if bar.clears(null_score):
        rejection = (
            f"the declared null model — the row's candidate pool in "
            f"deterministic shuffled order — scores {null_score:.4f}, which "
            f"clears the gate's own bar ({bar.text}). The corpus does not "
            f"separate the shipped ranker from a model that cannot represent "
            f"the distinction being measured"
        )
    # The verdict does not depend on the shipped score, so it is settled
    # before the shipped arm runs: a rejected corpus never drives
    # `retrieve()`, and an arm that raises still leaves a verdict behind.
    shipped_score, error = (None, None) if rejection else _measure(shipped)
    _verdict(
        module=module,
        family=Family.RANKING,
        shipped_score=shipped_score,
        null_score=null_score,
        bar=bar,
        rejection=rejection,
        extra=f"n_rows={len(rows)} {extra}".strip(),
        record_property=record_property,
        shipped_error=error,
    )
    assert shipped_score is not None
    return RankingResult(shipped=shipped_score, null=null_score)


def guard_classification_gate(
    *,
    module: str,
    rows: Sequence[Row],
    shipped: Callable[[], float],
    bar: Bar,
    score_constant: Callable[[str], float],
    record_property: RecordProperty,
    label_key: str = "label",
    extra: str = "",
) -> RankingResult:
    """Classification family: a constant predictor returning the majority label.

    `score_constant` is supplied by the gate because the rule is that
    the null is scored with *the gate's own* metric, and those differ —
    accuracy for most, a precision/recall headroom for two.
    """
    _assert_no_pool_shape(rows, module)
    null_label = majority_label(rows, label_key=label_key)
    null_score = score_constant(null_label)
    rejection = None
    if bar.clears(null_score):
        rejection = (
            f"a constant predictor answering {null_label!r} on every row "
            f"scores {null_score:.4f}, which clears the gate's own bar "
            f"({bar.text}). The corpus is imbalanced enough that the bar "
            f"measures the label distribution rather than the detector"
        )
    # Settled before the shipped arm runs, for the same reason as the
    # ranking guard: the verdict is about the corpus, not the detector.
    shipped_score, error = (None, None) if rejection else _measure(shipped)
    _verdict(
        module=module,
        family=Family.CLASSIFICATION,
        shipped_score=shipped_score,
        null_score=null_score,
        bar=bar,
        rejection=rejection,
        extra=f"n_rows={len(rows)} majority_label={null_label!r} {extra}".strip(),
        record_property=record_property,
        shipped_error=error,
    )
    assert shipped_score is not None
    return RankingResult(shipped=shipped_score, null=null_score)


def guard_ablation_gate(
    *,
    module: str,
    rows: Sequence[Row],
    arms: Callable[[], AblationArms],
    bar: Bar,
    record_property: RecordProperty,
    gold_key: str | None = None,
    pool_key: str | None = None,
    k_key: str = "k",
    default_k: int = 10,
    min_without_share: float = MIN_WITHOUT_SHARE,
    extra: str = "",
) -> AblationArms:
    """Ablation family: the ablated arm, plus the `without_rate > 0` share.

    The null model is the ablated arm. Its uplift against itself is
    identically zero by construction, so comparing that to the bar says
    nothing about the corpus — which is why the declared rejection is
    the share of rows the ablated arm scores above zero on. The bar
    comparison is still recorded (`null_clears_bar`), because a gate
    whose bar a zero uplift clears — any `>= 0` no-regression bar —
    carries no positive evidence about the feature, and a reader of the
    release record should see that rather than have to infer it.

    Omitting `gold_key`/`pool_key` turns the structural pre-filters off.
    That is a claim about the corpus, so `_assert_no_pool_shape` checks
    it against the rows: an ablation corpus that carries a candidate
    pool must wire the keys and let the filters run. This is the family
    where "gold == pool" matters most, because it is what produces the
    +1.000 uplift tautology the `without_rate` floor exists to reject.
    """
    if gold_key and pool_key:
        pre = structural_prefilter(
            rows,
            gold_key=gold_key,
            pool_key=pool_key,
            k_key=k_key,
            default_k=default_k,
        )
        if pre:
            _verdict(
                module=module,
                family=Family.ABLATION,
                shipped_score=None,
                null_score=0.0,
                bar=bar,
                rejection=(
                    f"structural pre-filter {pre.name!r} rejected it — "
                    f"{pre.detail}"
                ),
                extra=f"n_rows={len(rows)} prefilter={pre.name} {extra}".strip(),
                record_property=record_property,
            )
    elif gold_key or pool_key:
        raise ValueError("declare both gold_key and pool_key, or neither")
    else:
        _assert_no_pool_shape(rows, module)

    measured, error = _measure(arms)
    if error is not None:
        _verdict(
            module=module,
            family=Family.ABLATION,
            shipped_score=None,
            null_score=0.0,
            bar=bar,
            rejection=None,
            extra=f"n_rows={len(rows)} {extra}".strip(),
            record_property=record_property,
            shipped_error=error,
        )
    assert measured is not None
    if not measured.without_row_scores:
        raise ValueError("the ablated arm's per-row scores are required")
    share = sum(1 for s in measured.without_row_scores if s > 0) / len(
        measured.without_row_scores
    )
    rejection = None
    if share < min_without_share:
        rejection = (
            f"the ablated arm scores above zero on only {share:.1%} of "
            f"{len(measured.without_row_scores)} rows (floor "
            f"{min_without_share:.0%}). The feature is the only path to the "
            f"gold on this corpus, so the uplift the gate measures is a "
            f"tautology of the weight table rather than evidence about the "
            f"feature"
        )
    # Reported in the units the bar is stated in — uplift, not the raw
    # arm rates — so `shipped`, `null` and `bar` on the recorded line
    # are comparable. The null model's uplift against itself is 0.0 by
    # construction; the arms themselves ride along in `extra`.
    _verdict(
        module=module,
        family=Family.ABLATION,
        shipped_score=measured.uplift,
        null_score=0.0,
        bar=bar,
        rejection=rejection,
        extra=(
            f"n_rows={len(rows)} arm_with={measured.shipped:.4f} "
            f"arm_without={measured.ablated:.4f} "
            f"without_rate_share={share:.3f} "
            f"(floor {min_without_share:.2f}) {extra}"
        ).strip(),
        record_property=record_property,
    )
    return measured
