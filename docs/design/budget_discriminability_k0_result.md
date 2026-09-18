# K0 result: the injection budgets are unmeasurable at public corpus (#1546)

This is the K0 read-out. The rule it is read against is
`docs/design/budget_discriminability_decision_rule.md`, committed before any
number here existed.

No decision criterion in that rule was changed after these numbers were seen.
Two amendments were added after: Amendment 1 corrects a mechanism claim that
was false (admission is not monotone in budget), and Amendment 2 discloses
deviations the rule had not named. Both quote what they correct, and neither
touches EC, N, the grey band, the required N, the cut conditions, or the
finding wording. Read them before you read these numbers.

Re-derive every figure on this page with:

```
uv run python scripts/budget_discriminability_census.py
uv run python scripts/budget_discriminability_census.py --check
```

## The finding

> unmeasurable at N=7; reopen at belief-id-joined gold with per-query pools
> exceeding the smallest shipped budget

The census establishes N = 7 non-degenerate labelled queries over the two
public corpora.
<!-- derived: scripts/budget_discriminability_census.py#n = 7 -->

This is **not** "the budgets are correct". The census cannot say that, and the
rule forbids the phrasing.

## EC, on every measurable lane

EC = D / N is 0 on all six measurable lanes, on both public corpora.

| Lane | Symbol | Shipped budget | N | D | EC |
| --- | --- | --- | --- | --- | --- |
| `ups` | `hook.DEFAULT_HOOK_TOKEN_BUDGET` | 1500 | 7 | 0 | 0.0pp |
| `search_tool` | `hook_search_tool.INJECTED_TOKEN_BUDGET` | 600 | 7 | 0 | 0.0pp |
| `search_tool_bash` | `hook_search_tool.BASH_INJECTED_TOKEN_BUDGET` | 300 | 7 | 0 | 0.0pp |
| `agent_context` | `hook_agent_context.INJECTED_TOKEN_BUDGET` | 600 | 7 | 0 | 0.0pp |
| `retrieval_default` | `retrieval.DEFAULT_TOKEN_BUDGET` | 2400 | 7 | 0 | 0.0pp |
| `rebuilder` | `rebuild_log.DEFAULT_REBUILDER_TOKEN_BUDGET` | 4000 | 7 | 0 | 0.0pp |
<!-- derived: scripts/budget_discriminability_census.py#ec_pp.ups = 0.0 -->
<!-- derived: scripts/budget_discriminability_census.py#ec_pp.search_tool = 0.0 -->
<!-- derived: scripts/budget_discriminability_census.py#ec_pp.search_tool_bash = 0.0 -->
<!-- derived: scripts/budget_discriminability_census.py#ec_pp.agent_context = 0.0 -->
<!-- derived: scripts/budget_discriminability_census.py#ec_pp.retrieval_default = 0.0 -->
<!-- derived: scripts/budget_discriminability_census.py#ec_pp.rebuilder = 0.0 -->

Read that under the rule's exact clause: **EC = 0 means zero discordant pairs
are possible, not merely unobserved.** Every candidate pool in these corpora
prices below every budget in the grid, so both arms render byte-identical
blocks and every downstream metric is identical by construction. No sample size
over these corpora detects an effect, because there is no effect to detect at
any size. That is an impossibility result, not an underpowered null.

The mechanism shows up directly in `binds_on`: the cap that ended the arm is
`pool` on 7 of 7 queries, on every lane.
<!-- derived: scripts/budget_discriminability_census.py#pool_binds.ups = 7 -->

## N, and what it cost to get there

The two public labelled corpora carry 23 labelled queries between them —
16 from the inline corpus in `src/aelfrice/benchmark.py` and 7 from
`benchmarks/posterior_ranking/fixtures/default.jsonl`.
<!-- derived: scripts/budget_discriminability_census.py#labelled_queries_before_exclusion = 23 -->

16 of those are **degenerate** and are excluded: the retrieved pool is exactly
the gold set, so a truncating budget has nothing non-gold to drop and the query
cannot discriminate at any budget.
<!-- derived: scripts/budget_discriminability_census.py#degenerate_excluded = 16 -->

That leaves N = 7. The benchmark corpus's queries are built from 2-4 terms that
appear in exactly one belief, and FTS5 joins them with an implicit AND, so most
queries reach a pool of one. The pools that survive hold 2-4 beliefs and price
at 47-132 tokens across all six lanes, against a smallest shipped budget of
300. Read those two endpoints off the `pool_cost=[min,max]` columns of a full
run; the census does not emit them as figures.

Against the pre-registered required-N of 3,713 per arm for a 2.8pp effect near
a 74% base rate, N = 7 is short by more than two orders of magnitude — but that
comparison is beside the point here, because the ceiling is zero rather than
small.
<!-- derived: scripts/budget_discriminability_census.py#required_n_per_arm = 3713 -->

The grey band at this N is 37.0398pp, which is the binomial term swamping the
9.5pp inter-grader prior. It carries no A/A term, because K3 is not built.
<!-- derived: scripts/budget_discriminability_census.py#grey_band_pp = 37.0398 -->

## The one public cell where a budget could bind, and why it does not

A prior pass reported that the 16-belief corpus prices at about 359 tokens on
the `search_tool` lane, against the 300-token `Bash` budget, and called that the
one public cell where a budget can bind. Re-derived here under
`hook_search_tool._belief_line_cost`, the whole 16-belief store prices at **378
tokens**, which does exceed 300.
<!-- derived: scripts/budget_discriminability_census.py#bash_lane_benchmark_whole_store_cost = 378 -->

It is reported on its own row and not folded into EC, because no labelled query
reaches it. The largest pool any query actually retrieves on that lane prices at
50 tokens, and the number of labelled queries where the `Bash` budget binds is
0.
<!-- derived: scripts/budget_discriminability_census.py#bash_lane_benchmark_pool_cost_max = 50 -->
<!-- derived: scripts/budget_discriminability_census.py#bash_lane_benchmark_budget_binds = 0 -->

A budget that binds on a hypothetical whole-store pool has not bound on
anything a query produced.

## The lane this instrument cannot see

`hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET` is packed by
`hook._pack_core_candidates` over a queryless candidate set selected by
corroboration and posterior. It has no labelled-query population, so the census
reports no EC for it rather than a zero that would read as a measurement.

## What this run does not exercise

EC = 0 is measured on a narrower retrieval configuration than production. Three
narrowings, all printed by the census under "what this run does not exercise":

* **No edges.** The census seeds beliefs only, so the temporal spine, BFS
  expansion, and cluster structure are inert. Those are the sources where a
  budget could change candidacy rather than count.
* **The structural lane never fires.** `retrieval._route_structural_query` is
  default-on, but it answers only a `<KIND>:<target_id>` marker query and
  neither corpus holds one. It also prices with `retrieval._belief_tokens` and
  ignores `belief_cost_fn`, so a census over marker queries would need its own
  pricing.
* **The block ceiling is held fixed and never applied.** The decision rule
  calls a grid that holds `hook.HOOK_BLOCK_TOKEN_CEILING` fixed invalid. This
  census does hold it fixed and measures `retrieve()` output rather than a
  rendered hook block. At 6000 against a dearest pool of 132 tokens it cannot
  bind here, but the deviation is a deviation (decision rule, Amendment 2).

None of the three moves a number on these corpora. Each narrows the population
the zero is measured over, which is why it is published beside the zero.

## Bound status

The bound is **verified by mutation**, against three fake packers rather than
one. Each makes the census report violations, and the shipped packer reports
none:

| Fake packer | Violations |
| --- | --- |
| reads the budget as an item count | 402 |
| returns less at a higher budget | 1,035 |
| moves only a non-gold belief | 48 |
| shipped packer | 0 |

Those three counts are not published figures and the census does not emit them;
re-derive them by running the K2 arms in `tests/test_budget_census.py`. The 0 is
emitted.
<!-- derived: scripts/budget_discriminability_census.py#violations = 0 -->

Two of the three arms are new, because the first version of this guard could
not see them, and both misses mattered:

* The census reads its candidate pool by calling the same `retrieve()` path at
  a probe budget of 1e9, so a packer that returns *less* at a high budget shrank
  the pool that every footprint was compared against. It reported 0 violations,
  EC 0 on every lane, and exit 0 — and silently moved N from 7 to 23, because an
  empty pool passed the degeneracy test. The same fake confined below the probe
  budget reported 252. Every arm must now be a subset of the probe pool, and an
  empty pool is excluded from N and counted.
* The bound was checked on the gold footprint, while the sentence it licences
  is about the whole rendered block. A fake that moved only non-gold beliefs
  reported 0 violations on every lane. The bound is now checked on the block.

The shipped stage-2 skip belongs to the first of those families — see the
decision rule's Amendment 1 — so neither arm is hypothetical.

## What this does and does not license

* **No constant moves.** None moved on this branch.
* **A raise is not licensed.** A ceiling statistic says the arms could return
  different text, never which text is better, and a raise moves the pack in
  both directions: admission is not monotone, so raising a budget evicts
  beliefs as well as admitting them (decision rule, Amendment 1). EC is within
  the band and points nowhere, so even the "not excluded" sentence is not
  reached.
* **A cut is not licensed either.** The rule requires the cut arm's `binds_on`
  not to be `pool`, and it is `pool` on every cell; and it requires the latency
  arm, which is out of scope.
* **K1, K3 and the latency arm stay gated** on a further ruling.

## Reopen conditions

Reopen when a corpus offers labelled queries whose gold is joined by belief id
and whose per-query pools price above the smallest shipped budget of 300 tokens.
Neither public corpus does. LongMemEval-S may or may not; the claim that it is
saturated on recall is currently unsourced — see the #1546 CHANGELOG entry,
where the figure that carried that claim was struck for being un-re-derivable.
