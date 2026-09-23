#!/usr/bin/env python3
"""#1546 K0 — how many labelled queries could a budget change at all?

If the whole candidate pool for a labelled query prices below the smaller of two
budgets under that lane's shipped `belief_cost_fn`, neither budget can bind:
both arms render a byte-identical block, the model receives identical input, and
every downstream metric is identical by construction. The count of labelled
queries where the arms *could* differ is therefore an exact upper bound on any
achievable effect.

That is a **conditional** claim, and only that. A budget does not merely
truncate: `clustering.pack_with_clusters` skips an over-budget belief with
`continue`, so the budget selects rather than truncates and admission is not
monotone — raising a budget can evict a belief the lower budget admitted. The
bound above is untouched by that, because it applies only where no cap binds,
but any argument that reasons from monotonicity is void. See the decision
rule's section on the mechanism.

This script prices every candidate pool in the public labelled corpora against
every shipped budget, under each lane's real cost function, and reports

    EC = D / N

the share of non-degenerate labelled queries whose **gold-belief footprint in
the rendered lane block** differs between the shipped arm and a grid arm. The
pre-registered decision rule is `docs/design/budget_discriminability_decision_rule.md`.
Read it before reading any number here. EC is a ceiling on effect, never a
quality result.

## Usage

    uv run python scripts/budget_discriminability_census.py            # full run
    uv run python scripts/budget_discriminability_census.py --dry-run  # plan only
    uv run python scripts/budget_discriminability_census.py --check    # determinism
    uv run python scripts/budget_discriminability_census.py --emit-figures

`--dry-run` prints the lane table, the grid and the corpora without opening a
store or retrieving anything, and exits 0. `--check` runs the census twice and
exits non-zero unless the two reports hash identically. `--emit-figures` is the
protocol `scripts/check_derived_figures.py` speaks: a flat JSON object of
key -> value on stdout and nothing else. Every mode exits non-zero on failure,
and a bound violation is a failure.

## The bound, and how it can fail

The census asserts its own premise on every run, in two ways.

1. **Block identity.** For each query and each pair of grid arms whose smaller
   effective budget is at or above the pool's price, the two *whole rendered
   blocks* must be identical — not just the gold lines, because block identity
   is what the ceiling claim asserts.
2. **Probe containment.** Every arm's output must be a subset of the unbudgeted
   probe pool. The probe runs through the same `retrieve()` path as the arms, so
   a packer that returns *less* at a high budget would otherwise shrink the pool
   that its own detector is computed against and escape unseen. Containment is
   what makes the probe's shared code path safe to use.

A run where either fails exits 2 with the offending rows named.
`tests/test_budget_census.py` drives both cases against deliberately
budget-sensitive fake packers — one that truncates by item count, one that
returns less at a higher budget — because agreeing rows on unmodified code
prove nothing.

## Stores

Every store is `:memory:` and lives for one query. The script never opens the
user's store, never reads `AELFRICE_CORPUS_ROOT`, and never touches the lab
corpus (#1456). It clears every `AELFRICE_*` variable from its own environment
before measuring, because the lane resolvers are environment-first and
None-vs-False sensitive: a stray variable silently measures a different lane
set. The resolved lane set is printed into the report header either way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _clear_aelfrice_env() -> list[str]:
    """Drop every `AELFRICE_*` variable and report what was dropped.

    The precedent is `benchmarks/injection_budget_bytes.py`: a producer clears
    its own environment rather than relying on the caller, so a producer that
    stops clearing fails the derived-figures gate instead of being covered by
    it. Done before `aelfrice` is imported, because several resolvers read the
    environment at import time.
    """
    dropped = sorted(k for k in os.environ if k.startswith("AELFRICE_"))
    for key in dropped:
        del os.environ[key]
    return dropped


_DROPPED_ENV: Final[list[str]] = _clear_aelfrice_env()

from aelfrice import benchmark as bench  # noqa: E402
from aelfrice import hook, hook_agent_context, hook_search_tool  # noqa: E402
from aelfrice import rebuild_log, retrieval  # noqa: E402
from aelfrice.models import (  # noqa: E402
    BELIEF_FACTUAL,
    LOCK_NONE,
    ORIGIN_UNKNOWN,
    Belief,
)
from aelfrice.store import MemoryStore  # noqa: E402

# --- The grid ----------------------------------------------------------
#
# Pre-registered in docs/design/budget_discriminability_decision_rule.md.
# `DEFAULT_L25_TOKEN_SUBBUDGET` is varied because it is the cross-lane
# sub-cap between a budget and the L2.5 lane (#1546 prerequisite 2); a grid
# that held it fixed would measure the sub-cap rather than the budget.

BUDGET_MULTIPLIERS: Final[tuple[float, ...]] = (0.5, 0.75, 1.0, 1.5, 2.0)
L25_SUBBUDGETS: Final[tuple[int, ...]] = (200, 400, 800)
SHIPPED_MULTIPLIER: Final[float] = 1.0
SHIPPED_L25_SUBBUDGET: Final[int] = retrieval.DEFAULT_L25_TOKEN_SUBBUDGET

# A budget high enough that no cap can bind, used to read the unbudgeted
# candidate pool out of the same code path that the measured arms run.
POOL_PROBE_BUDGET: Final[int] = 10**9

# Power inputs, pre-registered. Not measured here; restated so the required
# sample size travels with the census that reports N.
POWER_BASE_RATE: Final[float] = 0.740
POWER_EFFECT: Final[float] = 0.028
POWER_Z_ALPHA: Final[float] = 1.959963984540054  # two-sided 0.05
POWER_Z_BETA: Final[float] = 0.8416212335729143  # power 0.80

# The #1255 proposal-17 inter-grader spread, a historical prior carried into
# the grey band. Not estimated here.
INTER_GRADER_SPREAD_PP: Final[float] = 9.5


# --- Lanes -------------------------------------------------------------


@dataclass(frozen=True)
class Lane:
    """One shipped injection lane, at the call shape it ships."""

    name: str
    symbol: str
    budget: int
    l1_limit: int
    cost_fn: Callable[[Belief], int] | None
    render: Callable[[Belief], str]
    manifest_reference_locks: bool


def _element_render(b: Belief) -> str:
    """The `<belief …>` element the hook lanes emit, one line."""
    return hook._belief_element_line(b)


def _search_tool_render(b: Belief) -> str:
    """The `[L0] <prefix>: <content>` line the Grep/Glob/Bash lane emits."""
    line = hook_search_tool._belief_line(b, frozenset())
    return "" if line is None else line


def _content_render(b: Belief) -> str:
    """The identity render, for lanes whose emitted artefact is the list.

    `retrieval.retrieve` and the v1.4 rebuilder hand a belief list to their
    callers rather than owning a line shape, so the rendered artefact for the
    footprint is the belief's own content. Naming it here keeps the census
    from inventing a line shape that no lane emits.
    """
    return f"{b.id}\t{b.content}"


def lanes() -> tuple[Lane, ...]:
    """The lanes this census can measure, at their shipped values.

    `hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET` is deliberately absent and
    that absence is a finding, not an oversight: the `<core>` section is packed
    by `hook._pack_core_candidates` over a queryless candidate set selected by
    corroboration and posterior, so it has no labelled-query population and this
    instrument cannot see it at all. See `report()['unmeasurable_lanes']`.
    """
    return (
        Lane(
            name="ups",
            symbol="hook.DEFAULT_HOOK_TOKEN_BUDGET",
            budget=hook.DEFAULT_HOOK_TOKEN_BUDGET,
            l1_limit=retrieval.DEFAULT_L1_LIMIT,
            cost_fn=hook._ups_belief_line_cost,
            render=_element_render,
            manifest_reference_locks=True,
        ),
        Lane(
            name="search_tool",
            symbol="hook_search_tool.INJECTED_TOKEN_BUDGET",
            budget=hook_search_tool.INJECTED_TOKEN_BUDGET,
            l1_limit=hook_search_tool.INJECTED_L1_LIMIT,
            cost_fn=hook_search_tool._belief_line_cost,
            render=_search_tool_render,
            manifest_reference_locks=True,
        ),
        Lane(
            name="search_tool_bash",
            symbol="hook_search_tool.BASH_INJECTED_TOKEN_BUDGET",
            budget=hook_search_tool.BASH_INJECTED_TOKEN_BUDGET,
            l1_limit=hook_search_tool.BASH_INJECTED_L1_LIMIT,
            cost_fn=hook_search_tool._belief_line_cost,
            render=_search_tool_render,
            manifest_reference_locks=True,
        ),
        Lane(
            name="agent_context",
            symbol="hook_agent_context.INJECTED_TOKEN_BUDGET",
            budget=hook_agent_context.INJECTED_TOKEN_BUDGET,
            l1_limit=hook_agent_context.INJECTED_L1_LIMIT,
            cost_fn=None,
            render=_element_render,
            manifest_reference_locks=True,
        ),
        Lane(
            name="retrieval_default",
            symbol="retrieval.DEFAULT_TOKEN_BUDGET",
            budget=retrieval.DEFAULT_TOKEN_BUDGET,
            l1_limit=retrieval.DEFAULT_L1_LIMIT,
            cost_fn=None,
            render=_content_render,
            manifest_reference_locks=False,
        ),
        Lane(
            name="rebuilder",
            symbol="rebuild_log.DEFAULT_REBUILDER_TOKEN_BUDGET",
            budget=rebuild_log.DEFAULT_REBUILDER_TOKEN_BUDGET,
            l1_limit=retrieval.DEFAULT_L1_LIMIT,
            cost_fn=None,
            render=_content_render,
            manifest_reference_locks=False,
        ),
    )


UNMEASURABLE_LANES: Final[dict[str, str]] = {
    "hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET": (
        "queryless: hook._pack_core_candidates packs a corroboration- and "
        "posterior-selected candidate set, so no labelled query exists for it "
        "and this instrument reports no EC for it at all"
    ),
}

# What the census does not exercise. Each entry narrows the population the EC
# figures are measured over, so each is published beside them rather than left
# for a reader to discover. An undisclosed narrowing is how a zero measured on
# a thin configuration gets read as a zero on production.
NOT_EXERCISED: Final[dict[str, str]] = {
    "retrieval._route_structural_query": (
        "is_hrr_structural_enabled() resolves True, but the lane fires only "
        "on a `<KIND>:<target_id>` marker query and neither corpus contains "
        "one. It also prices with retrieval._belief_tokens and "
        "lock_injection_tokens and ignores belief_cost_fn, so a lane whose "
        "cost_fn is cheaper than _belief_tokens would see this census "
        "understate the packer's price. Unexercised here, not mis-reported"
    ),
    "hook.enforce_block_ceiling": (
        "the decision rule calls a grid that holds HOOK_BLOCK_TOKEN_CEILING "
        "fixed invalid. This census holds it fixed and never applies it: it "
        "measures retrieve() output rather than a rendered hook block. The "
        "ceiling is 6000 and no pool here prices near it, so it cannot bind "
        "on these corpora — but the deviation is a deviation and is named"
    ),
    "graph edges": (
        "_open_store inserts beliefs and no edges, so has_edge_type is False "
        "for every type and the temporal spine never runs. BFS expansion "
        "does not run either, but for a different reason that is worth "
        "keeping separate: it is gated on the bfs_on flag, which is "
        "default-off, so it would be absent here even on an edge-bearing "
        "store. Those are the sources where a budget could change candidacy "
        "rather than count, so EC here is measured on a retrieval "
        "configuration narrower than production. The cluster packer is NOT "
        "in that set and "
        "was wrongly listed here: pack_with_clusters runs on every query of "
        "every cell with singleton clusters -- 2208 calls in a full run, all "
        "of them packing -- so the non-monotone stage-2 `continue` this "
        "module's docstring flags is live on every measured cell, not inert. "
        "What the absent edges hide is the consequence: the spine lane seeds "
        "from l1_packed[:DEFAULT_SPINE_SEED_COUNT], so on an edge-bearing "
        "store a lower budget can promote a different belief into the seed "
        "window and reach a neighbour the unbudgeted probe never saw. "
        "Containment is therefore a property of THIS corpus, not a theorem; "
        "tests/test_budget_census.py::"
        "test_containment_is_a_property_of_this_corpus_and_not_a_theorem "
        "exhibits an arm that escapes on shipped code"
    ),
}


# --- Corpora -----------------------------------------------------------


@dataclass(frozen=True)
class LabelledQuery:
    """One labelled query: a query string, a seeded pool, and a gold set."""

    corpus: str
    qid: str
    query: str
    gold_ids: frozenset[str]
    beliefs: tuple[Belief, ...]


def _belief(bid: str, content: str) -> Belief:
    return Belief(
        id=bid,
        content=content,
        content_hash=f"census_{bid}",
        alpha=1.0,
        beta=1.0,
        type=BELIEF_FACTUAL,
        lock_level=LOCK_NONE,
        locked_at=None,
        created_at="2026-01-01T00:00:00Z",
        last_retrieved_at=None,
        origin=ORIGIN_UNKNOWN,
    )


def _bench_corpus() -> list[LabelledQuery]:
    """`src/aelfrice/benchmark.py` — 16 beliefs, 16 queries, one gold each.

    The corpus is inline in that module rather than a data file, so it is read
    through the module's own private tuples and every query sees the whole
    16-belief store.
    """
    beliefs = tuple(
        _belief(entry.id, entry.content) for entry in bench._CORPUS
    )
    return [
        LabelledQuery(
            corpus="benchmark",
            qid=q.correct_id,
            query=q.query,
            gold_ids=frozenset({q.correct_id}),
            beliefs=beliefs,
        )
        for q in bench._QUERIES
    ]


POSTERIOR_FIXTURES: Final[Path] = (
    REPO_ROOT / "benchmarks" / "posterior_ranking" / "fixtures" / "default.jsonl"
)


def _posterior_corpus() -> list[LabelledQuery]:
    """`benchmarks/posterior_ranking/fixtures/default.jsonl`.

    One gold belief and four distractors per query, each query getting its own
    store. Noise order is the file's order: the harness shuffles under a seed,
    but a shuffle changes which beliefs a *truncating* budget drops, so the
    census takes the committed order and says so rather than adding a seed the
    decision rule never pre-registered.
    """
    out: list[LabelledQuery] = []
    with POSTERIOR_FIXTURES.open(encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            fixture: dict[str, Any] = json.loads(raw)
            fid = str(fixture["id"])
            known_id = f"{fid}_known"
            beliefs = [_belief(known_id, str(fixture["known_belief_content"]))]
            for i, nc in enumerate(fixture["noise_belief_contents"]):
                beliefs.append(_belief(f"{fid}_noise_{i}", str(nc)))
            out.append(
                LabelledQuery(
                    corpus="posterior_ranking",
                    qid=fid,
                    query=str(fixture["query"]),
                    gold_ids=frozenset({known_id}),
                    beliefs=tuple(beliefs),
                )
            )
    return out


def corpora() -> list[LabelledQuery]:
    return _bench_corpus() + _posterior_corpus()


# --- Measurement -------------------------------------------------------


@dataclass
class QueryResult:
    corpus: str
    qid: str
    lane: str
    pool_ids: list[str]
    pool_cost: int
    degenerate: bool
    empty_pool: bool
    shipped_budget: int
    binds_on: str
    discordant: bool
    violations: list[str] = field(default_factory=list)


def _open_store(q: LabelledQuery) -> MemoryStore:
    store = MemoryStore(":memory:")
    for b in q.beliefs:
        store.insert_belief(b)
    return store


def arm_budget_for(lane_budget: int, multiplier: float) -> int:
    """One grid arm's token budget, rounded the way the grid rounds it.

    Public so a test can assert against the grid the census actually
    walks. A test that reimplements `int(lane.budget * m)` agrees with
    this for the shipped multipliers and diverges on the next one that
    lands on a half, which is exactly the drift a guard on the grid
    exists to prevent.
    """
    return max(1, int(round(lane_budget * multiplier)))


def _retrieve(
    store: MemoryStore,
    lane: Lane,
    query: str,
    *,
    token_budget: int,
    l25_token_subbudget: int,
) -> list[Belief]:
    return retrieval.retrieve(
        store,
        query,
        token_budget=token_budget,
        l1_limit=lane.l1_limit,
        l25_token_subbudget=l25_token_subbudget,
        manifest_reference_locks=lane.manifest_reference_locks,
        belief_cost_fn=lane.cost_fn,
    )


def _cost(lane: Lane, b: Belief) -> int:
    if lane.cost_fn is not None:
        return lane.cost_fn(b)
    return retrieval._belief_tokens(b)


def _footprint(lane: Lane, hits: Sequence[Belief], gold: frozenset[str]) -> str:
    """The gold-belief footprint of a rendered block.

    The ordered rendered lines of the gold beliefs only. Not the whole block,
    because a budget moving a non-gold belief is not a gold-footprint change;
    and not the id list, because a per-line character cap changes what a gold
    belief says while leaving its id in place.
    """
    return "\n".join(lane.render(b) for b in hits if b.id in gold)


def _block(lane: Lane, hits: Sequence[Belief]) -> str:
    """The whole rendered block: every hit, in order, not just the gold ones.

    `_footprint` is the pre-registered statistic and stays the gold subset.
    This is what the *bound* is checked on, because the bound's claim is block
    identity — "both arms render a byte-identical block" — and a check that
    reads only the gold lines cannot see a budget that moves a non-gold
    belief. Checking the weaker thing and publishing the stronger claim is how
    a guard passes over the defect it exists to catch.
    """
    return "\n".join(lane.render(b) for b in hits)


def _binds_on(pool_cost: int, budget: int, l25_subbudget: int) -> str:
    """Which cap ended the arm: `pool`, `token_budget`, `l25_subbudget`, `both`.

    The same four-value legend `benchmarks/injection_budget_bytes.py` prints.
    `pool` means the arm ran out of candidates before it ran out of budget,
    which is the reading that makes a constant inert on that cell.
    """
    budget_binds = pool_cost > budget
    l25_binds = pool_cost > l25_subbudget and l25_subbudget < budget
    if budget_binds and l25_binds:
        return "both"
    if budget_binds:
        return "token_budget"
    if l25_binds:
        return "l25_subbudget"
    return "pool"


def measure_query(lane: Lane, q: LabelledQuery) -> QueryResult:
    """Price one query's pool on one lane and compare every grid arm.

    The unbudgeted pool comes out of the same `retrieve()` call shape at
    `POOL_PROBE_BUDGET`, so it is the pool the shipped code actually builds
    rather than a reconstruction of it. That sharing is also the probe's
    weakness, and the containment check below is what pays for it: a packer
    that returns *less* at a higher budget poisons its own detector, because
    every footprint is computed against a pool the same mutation shrank. Each
    arm is therefore required to be a subset of the probe pool, and a run
    where it is not is a failed run rather than a measured zero.
    """
    store = _open_store(q)
    try:
        pool = _retrieve(
            store,
            lane,
            q.query,
            token_budget=POOL_PROBE_BUDGET,
            l25_token_subbudget=POOL_PROBE_BUDGET,
        )
        pool_ids = [b.id for b in pool]
        pool_set = frozenset(pool_ids)
        pool_cost = sum(_cost(lane, b) for b in pool)
        gold_in_pool = pool_set & q.gold_ids
        empty_pool = not pool_ids
        degenerate = bool(pool_ids) and pool_set <= q.gold_ids

        shipped_budget = int(round(lane.budget * SHIPPED_MULTIPLIER))
        violations: list[str] = []

        def arm(budget: int, sub: int) -> tuple[str, str]:
            """Run one arm; return its gold footprint and its whole block."""
            hits = _retrieve(
                store,
                lane,
                q.query,
                token_budget=budget,
                l25_token_subbudget=sub,
            )
            outside = [b.id for b in hits if b.id not in pool_set]
            if outside:
                violations.append(
                    f"{lane.name}/{q.corpus}/{q.qid}: the arm at "
                    f"budget={budget},sub={sub} returned {outside}, which the "
                    f"unbudgeted probe at budget={POOL_PROBE_BUDGET} did not "
                    "return. The probe is then not an upper bound on the "
                    "arms, so every footprint on this query is measured "
                    "against the wrong pool and no EC computed from it means "
                    "anything"
                )
            return _footprint(lane, hits, gold_in_pool), _block(lane, hits)

        shipped_fp, shipped_block = arm(shipped_budget, SHIPPED_L25_SUBBUDGET)

        discordant = False
        for mult in BUDGET_MULTIPLIERS:
            arm_budget = arm_budget_for(lane.budget, mult)
            for sub in L25_SUBBUDGETS:
                if arm_budget == shipped_budget and sub == SHIPPED_L25_SUBBUDGET:
                    continue
                arm_fp, arm_block = arm(arm_budget, sub)
                if arm_fp != shipped_fp:
                    discordant = True
                # The bound: when neither arm's caps can bind on this pool,
                # the two *blocks* must be byte-identical. A difference here
                # is not a finding about budgets, it is a refutation of the
                # premise this whole instrument rests on. It is checked on
                # the whole block and not on the gold subset, because block
                # identity is what the ceiling claim asserts.
                floor = min(arm_budget, shipped_budget)
                sub_floor = min(sub, SHIPPED_L25_SUBBUDGET)
                if (
                    arm_block != shipped_block
                    and pool_cost <= floor
                    and pool_cost <= sub_floor
                ):
                    violations.append(
                        f"{lane.name}/{q.corpus}/{q.qid}: pool_cost="
                        f"{pool_cost} <= min(budget)={floor} and "
                        f"min(l25_subbudget)={sub_floor}, yet the rendered "
                        f"block differs between budget={shipped_budget},"
                        f"sub={SHIPPED_L25_SUBBUDGET} and budget="
                        f"{arm_budget},sub={sub}"
                    )
        return QueryResult(
            corpus=q.corpus,
            qid=q.qid,
            lane=lane.name,
            pool_ids=pool_ids,
            pool_cost=pool_cost,
            degenerate=degenerate,
            empty_pool=empty_pool,
            shipped_budget=shipped_budget,
            binds_on=_binds_on(pool_cost, shipped_budget, SHIPPED_L25_SUBBUDGET),
            discordant=discordant,
            violations=violations,
        )
    finally:
        store.close()


def price_whole_corpus(lane: Lane, queries: Sequence[LabelledQuery]) -> dict[str, Any]:
    """Price every seeded belief of the largest store in a corpus.

    The per-query pools this census measures are what a *query* reaches. A
    separate question is what the corpus would cost if one query ever reached
    all of it, and that is the number a prior pass reported for the 16-belief
    corpus on the `search_tool` lane. It is reported on its own row rather than
    folded into EC, because no labelled query in these corpora produces it: a
    budget that binds on a hypothetical pool has not bound on anything.
    """
    best: tuple[int, int, str] = (0, 0, "")
    for q in queries:
        total = sum(_cost(lane, b) for b in q.beliefs)
        if total > best[0]:
            best = (total, len(q.beliefs), q.qid)
    return {
        "whole_store_cost": best[0],
        "beliefs": best[1],
        "largest_store_query": best[2],
        "exceeds_shipped_budget": best[0] > lane.budget,
    }


def required_n(
    p1: float = POWER_BASE_RATE,
    effect: float = POWER_EFFECT,
) -> int:
    """Per-arm sample size for `effect` at `p1`, two-sided 0.05, power 0.80."""
    p2 = p1 + effect
    pbar = (p1 + p2) / 2.0
    qbar = 1.0 - pbar
    num = (
        POWER_Z_ALPHA * math.sqrt(2.0 * pbar * qbar)
        + POWER_Z_BETA * math.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    return math.ceil(num / (p1 - p2) ** 2)


#: The K3 A/A replicate's measured band, in percentage points (#1546).
#:
#: Supplied by `scripts/budget_discriminability_aa_replicate.py`, which
#: re-runs this census under seeded permutations of insertion order and
#: reports the widest per-cell spread. Pinned here rather than computed on
#: every census run because the replicate runs this census nine times and
#: would make the third term cost nine-fold what the figure it feeds costs.
#:
#: It is a literal because it is a measurement, and it is 0.0 because the
#: statistic did not move — NOT because no term exists. Those two were
#: indistinguishable before K3 and `AA_BAND_MEASURED` is what tells them
#: apart. Re-derive it, do not adjust it:
#:
#:     uv run python scripts/budget_discriminability_aa_replicate.py --emit-figures
#:
#: `tests/test_budget_aa_replicate.py` fails if this constant and the
#: replicate's own figure disagree, so the two cannot drift.
AA_BAND_PP: Final[float] = 0.0
AA_BAND_MEASURED: Final[bool] = True


def grey_band(n: int) -> float:
    """`NF = max(1.96 * sqrt(0.25 / N), 9.5pp, A/A_band)`, in percentage points.

    All three terms are present since K3 (#1546). The A/A term is
    `AA_BAND_PP`, measured rather than assumed — a zero that reads like a
    measured one is exactly what it is, and `AA_BAND_MEASURED` distinguishes
    that from the earlier state in which no term existed at all.

    On the committed corpora the binomial term dominates at every N this
    census reaches, so adding the A/A term moves no published number. That
    is a fact about this corpus, not a reason the term is decorative: the
    band is a sample range and is monotone non-decreasing in the replicate
    count, so it can only ever rise.
    """
    if n <= 0:
        return max(INTER_GRADER_SPREAD_PP, AA_BAND_PP)
    binomial_pp = 100.0 * POWER_Z_ALPHA * math.sqrt(0.25 / n)
    return max(binomial_pp, INTER_GRADER_SPREAD_PP, AA_BAND_PP)


def resolvers() -> dict[str, object]:
    """Every lane resolver this census's numbers depend on, as resolved.

    Printed into the header because these are environment-first and
    None-vs-False sensitive: a stray `AELFRICE_*` variable silently measures a
    different lane set, and a report that does not name them cannot be
    compared with another run.
    """
    return {
        "entity_index_enabled": retrieval.is_entity_index_enabled(),
        "use_intentional_clustering": (
            retrieval.resolve_use_intentional_clustering()
        ),
        "max_coverage_pack": retrieval.is_max_coverage_pack_enabled(),
        "bfs_enabled": retrieval.is_bfs_enabled(),
        "heat_kernel_enabled": retrieval.is_heat_kernel_enabled(),
        "hrr_expand_enabled": retrieval.is_hrr_expand_enabled(),
        "hrr_structural_enabled": retrieval.is_hrr_structural_enabled(),
        "temporal_spine_enabled": retrieval.is_temporal_spine_enabled(),
        "origin_tiebreak": retrieval.is_origin_tiebreak_enabled(),
        "fan_effect": retrieval.is_fan_effect_enabled(),
        "posterior_weight": retrieval.resolve_posterior_weight(),
        "cleared_aelfrice_env": _DROPPED_ENV,
    }


def report(queries: Sequence[LabelledQuery] | None = None) -> dict[str, Any]:
    """Run the census and return the whole report as a plain dict.

    `queries` defaults to `corpora()`, which is the only population this
    registration measures. It is a parameter so that the K3 A/A replicate,
    `scripts/budget_discriminability_aa_replicate.py`, can hand the same
    labelled queries back under a permuted insertion order and reuse this
    statistic rather than reimplementing it — a replicate that reimplements
    EC measures its own reimplementation. The default path is unchanged, and
    `--check` hashes the same report it hashed before the parameter existed.
    """
    queries = list(corpora() if queries is None else queries)
    lane_rows: dict[str, Any] = {}
    violations: list[str] = []
    for lane in lanes():
        results = [measure_query(lane, q) for q in queries]
        by_corpus: dict[str, Any] = {}
        for corpus in sorted({r.corpus for r in results}):
            rows = [r for r in results if r.corpus == corpus]
            kept = [r for r in rows if not r.degenerate and not r.empty_pool]
            d = sum(1 for r in kept if r.discordant)
            n = len(kept)
            by_corpus[corpus] = {
                "n": n,
                "degenerate_excluded": sum(1 for r in rows if r.degenerate),
                "empty_pool_excluded": sum(1 for r in rows if r.empty_pool),
                "d": d,
                "ec_pp": round(100.0 * d / n, 4) if n else None,
                "pool_cost_min": min((r.pool_cost for r in kept), default=0),
                "pool_cost_max": max((r.pool_cost for r in kept), default=0),
                "pool_size_min": min(
                    (len(r.pool_ids) for r in kept), default=0
                ),
                "pool_size_max": max(
                    (len(r.pool_ids) for r in kept), default=0
                ),
                "binds_on": {
                    key: sum(1 for r in kept if r.binds_on == key)
                    for key in ("pool", "token_budget", "l25_subbudget", "both")
                },
                "queries_where_budget_can_bind": sorted(
                    r.qid for r in kept if r.binds_on != "pool"
                ),
                "whole_corpus": price_whole_corpus(
                    lane, [q for q in queries if q.corpus == corpus]
                ),
            }
        for r in results:
            violations.extend(r.violations)
        kept_all = [
            r for r in results if not r.degenerate and not r.empty_pool
        ]
        n_all = len(kept_all)
        d_all = sum(1 for r in kept_all if r.discordant)
        lane_rows[lane.name] = {
            "symbol": lane.symbol,
            "shipped_budget": lane.budget,
            "l1_limit": lane.l1_limit,
            "cost_fn": (
                f"{lane.cost_fn.__module__}.{lane.cost_fn.__name__}"
                if lane.cost_fn is not None
                else "retrieval._belief_tokens (default)"
            ),
            "n": n_all,
            "d": d_all,
            "ec_pp": round(100.0 * d_all / n_all, 4) if n_all else None,
            "by_corpus": by_corpus,
        }

    total_n = max((row["n"] for row in lane_rows.values()), default=0)
    return {
        "instrument": "scripts/budget_discriminability_census.py",
        "decision_rule": (
            "docs/design/budget_discriminability_decision_rule.md"
        ),
        "statistic": "EC = D / N, an exact ceiling on effect, never a quality result",
        "resolvers": resolvers(),
        "grid": {
            "budget_multipliers": list(BUDGET_MULTIPLIERS),
            "l25_subbudgets": list(L25_SUBBUDGETS),
            "shipped_cell": [SHIPPED_MULTIPLIER, SHIPPED_L25_SUBBUDGET],
            "pool_probe_budget": POOL_PROBE_BUDGET,
        },
        "constants": {
            "hook.DEFAULT_HOOK_TOKEN_BUDGET": hook.DEFAULT_HOOK_TOKEN_BUDGET,
            "hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET": (
                hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET
            ),
            "hook.HOOK_BLOCK_TOKEN_CEILING": hook.HOOK_BLOCK_TOKEN_CEILING,
            "hook_search_tool.INJECTED_TOKEN_BUDGET": (
                hook_search_tool.INJECTED_TOKEN_BUDGET
            ),
            "hook_search_tool.BASH_INJECTED_TOKEN_BUDGET": (
                hook_search_tool.BASH_INJECTED_TOKEN_BUDGET
            ),
            "hook_agent_context.INJECTED_TOKEN_BUDGET": (
                hook_agent_context.INJECTED_TOKEN_BUDGET
            ),
            "retrieval.DEFAULT_TOKEN_BUDGET": retrieval.DEFAULT_TOKEN_BUDGET,
            "retrieval.DEFAULT_L25_TOKEN_SUBBUDGET": (
                retrieval.DEFAULT_L25_TOKEN_SUBBUDGET
            ),
            "rebuild_log.DEFAULT_REBUILDER_TOKEN_BUDGET": (
                rebuild_log.DEFAULT_REBUILDER_TOKEN_BUDGET
            ),
        },
        "unmeasurable_lanes": UNMEASURABLE_LANES,
        "not_exercised": NOT_EXERCISED,
        "lanes": lane_rows,
        "n": total_n,
        "grey_band_pp": round(grey_band(total_n), 4),
        "grey_band_has_aa_term": AA_BAND_MEASURED,
        "required_n_per_arm": required_n(),
        "violations": violations,
    }


def figures(rep: dict[str, Any] | None = None) -> dict[str, Any]:
    """The flat key -> value object `--emit-figures` publishes."""
    rep = rep if rep is not None else report()
    out: dict[str, Any] = {
        "n": rep["n"],
        "required_n_per_arm": rep["required_n_per_arm"],
        "grey_band_pp": rep["grey_band_pp"],
        "violations": len(rep["violations"]),
    }
    first_lane = next(iter(rep["lanes"].values()))
    out["degenerate_excluded"] = sum(
        c["degenerate_excluded"] for c in first_lane["by_corpus"].values()
    )
    out["empty_pool_excluded"] = sum(
        c["empty_pool_excluded"] for c in first_lane["by_corpus"].values()
    )
    out["labelled_queries_before_exclusion"] = (
        out["degenerate_excluded"] + out["empty_pool_excluded"] + rep["n"]
    )
    for name, row in rep["lanes"].items():
        out[f"ec_pp.{name}"] = row["ec_pp"]
        out[f"pool_binds.{name}"] = sum(
            c["binds_on"]["pool"] for c in row["by_corpus"].values()
        )
    bench_bash = rep["lanes"]["search_tool_bash"]["by_corpus"]["benchmark"]
    out["bash_lane_benchmark_pool_cost_max"] = bench_bash["pool_cost_max"]
    out["bash_lane_benchmark_budget_binds"] = (
        bench_bash["binds_on"]["token_budget"] + bench_bash["binds_on"]["both"]
    )
    out["bash_lane_benchmark_whole_store_cost"] = bench_bash["whole_corpus"][
        "whole_store_cost"
    ]
    return out


# --- Output ------------------------------------------------------------


def render_text(rep: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("#1546 K0 — budget discriminability census")
    lines.append(f"instrument   {rep['instrument']}")
    lines.append(f"decision rule {rep['decision_rule']}")
    lines.append(f"statistic    {rep['statistic']}")
    lines.append("")
    lines.append("resolvers (env-first; a stray AELFRICE_* changes the lane set)")
    for key in sorted(rep["resolvers"]):
        lines.append(f"  {key:<32} {rep['resolvers'][key]!r}")
    lines.append("")
    lines.append("constants, read at their definition sites")
    for key in sorted(rep["constants"]):
        lines.append(f"  {key:<48} {rep['constants'][key]}")
    lines.append("")
    lines.append(
        f"grid  budgets x{rep['grid']['budget_multipliers']} "
        f"l25_subbudget {rep['grid']['l25_subbudgets']} "
        f"shipped cell {rep['grid']['shipped_cell']}"
    )
    lines.append("")
    for name in sorted(rep["lanes"]):
        row = rep["lanes"][name]
        lines.append(
            f"lane {name}  budget={row['shipped_budget']} "
            f"l1_limit={row['l1_limit']} cost_fn={row['cost_fn']}"
        )
        lines.append(f"  symbol {row['symbol']}")
        for corpus in sorted(row["by_corpus"]):
            c = row["by_corpus"][corpus]
            lines.append(
                f"  {corpus:<18} N={c['n']:<3} D={c['d']:<3} "
                f"EC={c['ec_pp']}pp  degenerate_excluded={c['degenerate_excluded']}"
                f"  empty_pool_excluded={c['empty_pool_excluded']}"
                f"  pool_cost=[{c['pool_cost_min']},{c['pool_cost_max']}]"
                f"  pool_size=[{c['pool_size_min']},{c['pool_size_max']}]"
                f"  binds_on={c['binds_on']}"
            )
            if c["queries_where_budget_can_bind"]:
                lines.append(
                    "    budget can bind on: "
                    + ", ".join(c["queries_where_budget_can_bind"])
                )
            w = c["whole_corpus"]
            lines.append(
                f"    whole store ({w['beliefs']} beliefs) prices at "
                f"{w['whole_store_cost']} tokens; exceeds the shipped "
                f"budget: {w['exceeds_shipped_budget']} — hypothetical, no "
                "labelled query reaches it"
            )
        lines.append(f"  lane total  N={row['n']} D={row['d']} EC={row['ec_pp']}pp")
        lines.append("")
    lines.append("lanes this instrument cannot see")
    for key, why in sorted(rep["unmeasurable_lanes"].items()):
        lines.append(f"  {key}: {why}")
    lines.append("")
    lines.append("what this run does not exercise")
    for key, why in sorted(rep["not_exercised"].items()):
        lines.append(f"  {key}: {why}")
    lines.append("")
    lines.append(f"N                  {rep['n']}")
    lines.append(
        f"grey band NF       {rep['grey_band_pp']}pp "
        f"(A/A term present: {rep['grey_band_has_aa_term']})"
    )
    lines.append(f"required N per arm {rep['required_n_per_arm']}")
    lines.append(f"bound violations   {len(rep['violations'])}")
    for v in rep["violations"]:
        lines.append(f"  VIOLATION {v}")
    return "\n".join(lines) + "\n"


def render_dry_run() -> str:
    lines = ["#1546 K0 census — dry run, nothing retrieved, no store opened", ""]
    lines.append("lanes")
    for lane in lanes():
        lines.append(
            f"  {lane.name:<18} budget={lane.budget:<5} "
            f"l1_limit={lane.l1_limit:<3} symbol={lane.symbol}"
        )
    lines.append("")
    lines.append("lanes this instrument cannot see")
    for key, why in sorted(UNMEASURABLE_LANES.items()):
        lines.append(f"  {key}: {why}")
    lines.append("")
    lines.append("what this run does not exercise")
    for key, why in sorted(NOT_EXERCISED.items()):
        lines.append(f"  {key}: {why}")
    lines.append("")
    qs = corpora()
    lines.append("corpora")
    for corpus in sorted({q.corpus for q in qs}):
        rows = [q for q in qs if q.corpus == corpus]
        lines.append(
            f"  {corpus:<18} queries={len(rows)} "
            f"beliefs_per_store=[{min(len(q.beliefs) for q in rows)},"
            f"{max(len(q.beliefs) for q in rows)}]"
        )
    lines.append("")
    cells = len(BUDGET_MULTIPLIERS) * len(L25_SUBBUDGETS)
    lines.append(
        f"grid cells per query per lane: {cells} "
        f"({len(BUDGET_MULTIPLIERS)} budget multipliers x "
        f"{len(L25_SUBBUDGETS)} l25 sub-budgets), 1 of them the shipped cell"
    )
    lines.append(
        f"retrieve() calls: {len(qs) * len(lanes()) * (cells + 1)} "
        "(one pool probe and one shipped arm per query per lane, plus the grid)"
    )
    lines.append(f"required N per arm (pre-registered): {required_n()}")
    return "\n".join(lines) + "\n"


def _stable_hash(rep: dict[str, Any]) -> str:
    payload = json.dumps(rep, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and exit without opening a store",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="run twice and exit non-zero unless both reports hash identically",
    )
    ap.add_argument(
        "--emit-figures",
        action="store_true",
        help="emit a flat JSON object of key -> value on stdout and nothing else",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="emit the whole report as JSON instead of text",
    )
    args = ap.parse_args(argv)

    if args.dry_run:
        sys.stdout.write(render_dry_run())
        return 0

    if args.check:
        first = report()
        second = report()
        h1, h2 = _stable_hash(first), _stable_hash(second)
        if h1 != h2:
            print(
                f"FAIL: census is not deterministic ({h1} != {h2})",
                file=sys.stderr,
            )
            return 1
        if first["violations"]:
            for v in first["violations"]:
                print(f"FAIL: bound violation: {v}", file=sys.stderr)
            return 2
        print(f"OK: byte-identical across two runs, sha256 {h1}")
        return 0

    rep = report()
    if args.emit_figures:
        json.dump(figures(rep), sys.stdout, sort_keys=True)
        sys.stdout.write("\n")
        return 2 if rep["violations"] else 0
    if args.json:
        json.dump(rep, sys.stdout, sort_keys=True, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_text(rep))
    return 2 if rep["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
