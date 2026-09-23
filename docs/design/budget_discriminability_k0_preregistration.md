# K0 pre-registration: the budget discriminability census (#1546)

This document is the live pre-registration for K0 of the injection-budget
re-tune ([#1546](https://github.com/robotrocketscience/aelfrice/issues/1546)).
Read it before you read any number the census prints.

It replaces `docs/design/budget_discriminability_decision_rule.md`, which the
operator voided on 2026-09-23. That rule described the instrument wrongly in
four places, so the verdict it licensed could be read as stronger than the
instrument supports. The four corrections are in
[The instrument's disclosed limits](#the-instruments-disclosed-limits), and
they are the substantive difference between the two registrations.

## What this registration carries forward, and what it adds

Every decision criterion is carried forward unchanged: the statistic, the
population rule, the grey-band construction, the required N, the three cut
conditions, the forbidden fallbacks, and the finding wording all say what the
voided rule said. Nothing is narrowed, widened, or re-thresholded.

That constraint is deliberate, and you should know why. This registration is
written by an author who has already seen the earlier run's figures, so it
cannot claim its criteria were chosen blind. Holding every threshold fixed is
what keeps the re-derivation honest: a criterion that moved could have been
fitted to the numbers, and a criterion that did not move could not. The
additions are disclosures and kill criteria, which can only make a verdict
weaker, never stronger.

## The question K0 answers

Over the two public labelled corpora, is there at least one labelled query for
which two token budgets in the pre-registered grid produce different
gold-belief footprints in a lane's rendered block?

The answer is falsifiable in one direction and only that direction. A "no" is
refuted by exhibiting a single lane, corpus, query, and pair of grid cells
whose footprints differ. A "yes" is refuted by re-running the census and
finding the pair agrees. The census answers by counting those queries, so the
question is decided by a count and not by a judgment.

K0 does not ask whether the shipped budgets are correct, whether a different
budget retrieves better beliefs, or whether any budget should move. Those are
quality questions and this instrument cannot reach them. See
[What a verdict can and cannot license](#what-a-verdict-can-and-cannot-license).

## The population

### What is in

The population is every labelled query in the two public corpora that the
census reads:

* `src/aelfrice/benchmark.py` — an inline corpus of 16 beliefs (`bench._CORPUS`)
  and 16 queries (`bench._QUERIES`), one gold belief per query. Every query
  sees the whole 16-belief store.
* `benchmarks/posterior_ranking/fixtures/default.jsonl` — 7 fixtures, each one
  query with 1 gold belief and 4 distractors, each in its own 5-belief store.

That is 23 labelled queries before exclusion. Re-derive the two counts from
the sources rather than from this document:

```
uv run python -c "import sys; sys.path.insert(0,'src'); from aelfrice import benchmark as b; print(len(b._CORPUS), len(b._QUERIES))"
grep -c . benchmarks/posterior_ranking/fixtures/default.jsonl
```

Noise order is the committed order of the fixture file. The posterior-ranking
harness shuffles under a seed, but a shuffle changes which beliefs a truncating
budget drops, so the census takes the file order and says so rather than
introducing a seed this registration never named (#605).

### What is out, and the rule for each exclusion

A query leaves the population under exactly one of two rules, both applied to
the unbudgeted probe pool that `measure_query` reads at
`POOL_PROBE_BUDGET = 1e9`:

1. **Degenerate.** The pool is non-empty and every belief in it is gold
   (`pool_set <= q.gold_ids`). A budget can only remove beliefs, so when every
   candidate is gold no budget can change which gold beliefs are present. The
   query cannot discriminate at any budget and is excluded, counted, and
   reported as `degenerate_excluded`.
2. **Empty pool.** The probe returns nothing (`not pool_ids`). The empty set is
   a subset of the gold set, so without this rule a vanished pool would pass
   the degeneracy test and count in N while contributing nothing to D. Excluded,
   counted, and reported as `empty_pool_excluded`.

`N` is what survives both rules. Both counts are outcomes of the run, not
inputs to it, so neither appears in this registration; the census reports them
per lane and per corpus, and the result document records them.

## The statistic

**EC = D / N**, reported in percentage points, per lane and per corpus.

* `N` ranges over the non-degenerate, non-empty labelled queries defined above,
  for one lane.
* `D` ranges over that same set, and counts the queries whose **gold-belief
  footprint in the rendered lane block** differs between the shipped arm and at
  least one other cell of the grid.

The footprint is the ordered sequence of rendered lines for the gold beliefs,
taken from the lane's own renderer. It is not the belief content, and it is not
the whole block: a per-line character cap can change what a gold belief says
while leaving its id in place, and a budget that moves a non-gold belief is not
a gold-footprint change.

The grid is pre-registered:

| Axis | Values |
| --- | --- |
| Budget multipliers | 0.5, 0.75, 1.0, 1.5, 2.0 |
| `retrieval.DEFAULT_L25_TOKEN_SUBBUDGET` | 200, 400, 800 |
| Shipped cell | multiplier 1.0, sub-budget 400 |

The L2.5 sub-cap varies because it sits between a budget and the L2.5 lane. A
grid that holds it fixed measures the sub-cap, not the budget.

### The exact clause

**EC = 0 means zero discordant pairs are possible, not merely unobserved.** If
every candidate pool prices below the smaller of two budgets, both arms render
byte-identical blocks, the model receives identical input, and every downstream
metric is identical by construction. No sample size over these corpora detects
an effect, because there is no effect to detect at any size. Report that as an
impossibility result. Reporting it as an underpowered null is wrong: an
underpowered null says you could not see it, and this says it is not there to
see.

EC is a ceiling on effect and nothing else. It is a change statistic in the
sense `benchmarks/README.md` defines. Never report it as a quality result, an
uplift, or a regression.

## The decision rule

### The grey band

`NF = max(1.96 * sqrt(0.25 / N), 9.5pp, A/A_band)`, in percentage points.

* The first term is the two-sided normal half-width for a proportion at its
  worst-case variance, at the N the census establishes.
* The 9.5pp term is the #1255 proposal-17 inter-grader spread. It is a
  historical prior carried forward, not an estimate produced here. An effect
  smaller than the disagreement between two human graders on the same answers
  is not a result this repo acts on.
* `A/A_band` is the K3 replicate's observed spread.

**What the band does not contain.** K3 is not built, so there is no A/A term.
The census sets `grey_band_has_aa_term` to `False` rather than substituting a
zero that would read like a measured one. The band therefore covers grader
disagreement and binomial sampling noise, and covers no instrument noise at
all: it cannot tell you how much of a non-zero EC is the census disagreeing
with itself.

You may not narrow the band after you see a number. If K3 later widens it, the
wider band applies **retroactively** to any verdict read against this
registration, including this one.

### What each EC value licenses

| Observation | Licensed action |
| --- | --- |
| `EC <= NF` on every lane | No constant moves. Report the finding wording below. |
| `EC > NF` and the effect points up | Publish exactly one sentence: "A raise is not excluded; referred to an answer-quality gate that does not exist." Move no constant. |
| `EC <= NF` on a cut arm | A cut is licensed only when all three conditions below hold. |

A cut arm needs all three:

1. its EC is within `NF`;
2. its `binds_on` is not `pool` — a cut whose binding cap is the pool changed
   nothing, and licensing it would license a no-op as a finding;
3. latency does not worsen.

Condition 3 needs the latency arm, which does not exist. K0 alone therefore
cannot license a cut either. K0 can only tell you whether the question is
answerable at all.

### The forbidden fallbacks

None of these may stand in for a quality verdict, in the published finding or
in any follow-up summary: injected bytes or any byte count; item counts or "N
of M queries changed"; top-k movement rate; Jaccard overlap at k; Kendall tau.
This repo has already retracted a published table for treating a change
statistic as quality evidence — see `benchmarks/README.md`, "What these
harnesses measure, and what they do not", and commit `848dbf83`.

### The finding wording

> unmeasurable at N=&lt;n&gt;; reopen at belief-id-joined gold with per-query pools
> exceeding the smallest shipped budget

Never "the budgets are correct". The census cannot say that. It can say only
that these corpora cannot tell the arms apart.

### One constant does not move regardless of outcome

`rebuild_log.DEFAULT_REBUILDER_TOKEN_BUDGET` is frozen by this registration. A
user can set `[rebuilder] token_budget` in `.aelfrice.toml`, so changing the
default silently reinterprets a value someone already wrote against the old
meaning. Moving it is a separate decision with its own migration question, not
a by-product of a re-tune.

## Kill criteria

Each criterion below is a threshold on a quantity the instrument computes and
prints. When one fires, the run is **uninformative**: it produces no verdict,
and you fix the instrument and re-run rather than reading EC.

Criteria are split by whether these corpora can supply the input that trips
them. That split is the point. A criterion listed as a guard when it cannot
fire is how a reader comes to believe a run was checked for something it was
never checked for.

### Criteria that can fire on these corpora

| # | Quantity | Threshold that kills the run |
| --- | --- | --- |
| K-1 | `report()['n']`, per lane | `n == 0`, so `ec_pp` is `None` and there is no statistic. The population rules already remove most of the 23 queries, so this is the criterion the corpora come closest to tripping. |
| K-2 | `empty_pool_excluded`, summed over corpora | `> 0`. The probe returned nothing for a labelled query, which means the run is measuring candidate generation or the gold join rather than the budget. |
| K-3 | The two `sha256` hashes that `--check` compares | They differ. The census is then not deterministic and no figure from it is re-derivable (#605). |
| K-4 | `report()['resolvers']` | Any key resolves differently from the set this registration records below. The lane set measured is then not the lane set registered. The census clears every `AELFRICE_*` variable from its own process, so a stray environment variable cannot trip this; a changed default in `src/` can. |
| K-5 | `len(report()['violations'])` | `> 0`. Either the block-identity check or the probe-containment check failed, so the premise the ceiling rests on is refuted and no EC computed under it means anything. The census exits 2. |
| K-6 | The exit status of `uv run pytest tests/test_budget_census.py` | Non-zero. The bound is then unverified and every number the census prints inherits that caveat. |

The resolver set K-4 is registered against, read on an environment the census
has cleared:

| Resolver | Registered value |
| --- | --- |
| `entity_index_enabled` | `True` |
| `use_intentional_clustering` | `True` |
| `max_coverage_pack` | `False` |
| `bfs_enabled` | `False` |
| `heat_kernel_enabled` | `False` |
| `hrr_expand_enabled` | `False` |
| `hrr_structural_enabled` | `True` |
| `temporal_spine_enabled` | `True` |
| `origin_tiebreak` | `False` |
| `fan_effect` | `False` |
| `posterior_weight` | `0.5` |

K-5 deserves one more line, because its zero is the easiest thing on this page
to misread. The check runs on every arm of every cell, and the K2 mutation arms
in `tests/test_budget_census.py` establish that it fires when the premise
breaks. Its power comes from those mutations, never from its own zero.

### Criteria that cannot fire on these corpora

State these as limits, not as guards.

* **A probe-containment violation on shipped code.** `_open_store` inserts
  beliefs and no edges, so `store.has_edge_type` is `False` for all ten edge
  types. `retrieval.retrieve` gates the temporal spine on
  `store.has_edge_type(EDGE_TEMPORAL_NEXT)`, and BFS is gated on `bfs_on`,
  which `is_bfs_enabled` resolves from the default-off `[retrieval]
  bfs_enabled` flag, so neither lane runs. The candidate sources that
  remain do not gain members as a budget falls. On an edge-bearing corpus this
  criterion becomes live, and
  `test_containment_is_a_property_of_this_corpus_and_not_a_theorem` exhibits an
  arm that escapes on shipped code once one `TEMPORAL_NEXT` edge exists.
* **`EC > NF`, if every candidate pool prices below the smallest arm budget in
  the grid.** This is a conditional, registered before the run: when it holds,
  no cap binds anywhere in the grid, every arm returns the whole pool, `D` is
  0 by construction, and the raise branch of the decision rule is unreachable.
  That outcome is **not** a kill. It is the impossibility result the exact
  clause describes, and you report it as the finding.
* **The cut branch, in full.** Condition 3 needs the latency arm, which is out
  of scope for K0 and does not exist in the tree. No K0 run can license a cut,
  whatever the numbers.

## What voids the run

A voided run is worse than an uninformative one: its numbers may not be
published at all, and the replacement is a fresh registration and a fresh run.

* You change any criterion in this document after seeing a census number.
* The resolved lane set differs from the table above (this is both K-4 and a
  void, because a changed lane set means the registered population was never
  measured).
* The census script, either corpus, or `clustering.pack_with_clusters` changes
  between the `--emit-figures` run and the `--check` run.
* The census runs with any `AELFRICE_*` variable it did not clear itself, or
  against any store other than the `:memory:` stores it builds. It must never
  open a user store, read `AELFRICE_CORPUS_ROOT`, or touch the lab corpus
  (#1456).
* A figure is published without a producer committed to the tree (#1469).

## The instrument's disclosed limits

This section is why the earlier registration was voided. Each item states
something the census does **not** establish, so no later reader can take the
verdict to be stronger than the instrument.

### 1. The cluster packer is live on every measured cell

The voided registration listed cluster structure among the things an edgeless
store makes inert. That was false. `clustering.pack_with_clusters` runs on
every query of every cell, with singleton clusters, and every one of those
calls packs. Count them:

```
uv run python scripts/budget_discriminability_census.py --dry-run
```

The dry run prints the `retrieve()` call count, and `pack_with_clusters` runs
once per `retrieve()` call. So the non-monotone stage-2 fill — stage 1 abandons
on the first representative it cannot afford, and stage 2 then walks the rest
of the pool with `continue` — is live on every cell this census measures. Any
argument that reasons from "cluster structure is inert here" is void, and any
argument that reasons from monotone admission is void with it:
`test_raising_a_budget_can_evict_a_belief_a_lower_budget_admitted` shows a
two-belief store priced at 63 and 19 tokens where raising `token_budget` from
62 to 63 **evicts** the belief the lower budget admitted.

### 2. Containment is a property of these corpora, not a theorem

The census reads its candidate pool by calling the same `retrieve()` path at a
1e9 probe budget, and requires every arm's output to be a subset of that pool.
That containment holds here **only because `_open_store` writes no edges**.

Add one `TEMPORAL_NEXT` edge and an arm at a *lower* budget reaches a belief
the probe did not: the temporal spine seeds from
`l1_packed[:DEFAULT_SPINE_SEED_COUNT]`, and `l1_packed` at a low budget is not
a prefix of `l1_packed` at a high one, because of the stage-2 skip in item 1. A
cheap belief promoted into the seed window reaches a neighbour that shares no
term with the query, so no L1 or L2.5 hit can account for it.
`test_containment_is_a_property_of_this_corpus_and_not_a_theorem` exhibits that
on shipped code and reds under a monotone-truncation mutation of
`pack_with_clusters`.

Read the containment zero as "these corpora cannot produce an escape", never as
"no arm can escape".

### 3. The containment mutation guard had no power on three of six lanes

Until #1546, the K2 fake that exercises the containment check fired at
`token_budget >= 700`. The entire arm grids of `ups` (750–3000),
`retrieval_default` (1200–4800), and `rebuilder` (2000–8000) sit above that
threshold, so on those three lanes the fake shrank the probe and every arm by
the same item, the comparison stayed consistent, and no violation could arise.
The assertion read the pooled violation list, so the three lanes the fake could
not reach were carried by the three it could, and their containment guard was
unmutation-tested while reading as covered.

The threshold now sits above every arm budget in the grid and far below the
probe budget, the test asserts that placement against
`census.arm_budget_for` rather than a reimplementation of it, and the assertion
is per lane and per violation type. Treat any K2 result from before that fix as
covering three lanes, not six.

### 4. BFS is absent because of a flag, not because of the corpora

`is_bfs_enabled` resolves `False` from the default-off `[retrieval]
bfs_enabled` flag, not from the absent edges. BFS expansion would be absent on
an edge-bearing store too. The
voided registration folded this into the edges disclosure, which made the
narrowing look corpus-specific when it is configuration-specific. A corpus with
edges closes item 2 and does not close this one.

### 5. Three further narrowings, carried forward

These stood in the voided registration and still stand:

* **No edges.** The temporal spine never runs, so the census never exercises
  the one source where a budget changes candidacy rather than count.
* **The structural lane never fires.** `retrieval._route_structural_query`
  resolves on, but answers only a `<KIND>:<target_id>` marker query, and
  neither corpus holds one. It also prices with `retrieval._belief_tokens` and
  ignores `belief_cost_fn`, so a census over marker queries would need its own
  pricing.
* **The block ceiling is held fixed and never applied.** The voided rule calls
  a grid that holds `hook.HOOK_BLOCK_TOKEN_CEILING` fixed invalid. This census
  holds it fixed and never calls `hook.enforce_block_ceiling` at all: it
  measures `retrieve()` output, not a rendered hook block. The deviation is
  disclosed, not repaired.

### 6. One lane this instrument cannot see at all

`hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET` is packed by
`hook._pack_core_candidates` over a queryless candidate set selected by
corroboration and posterior. It has no labelled-query population, so the census
reports no EC for it rather than a zero that would read as a measurement. Seven
constants are under adjudication and this instrument reaches six of them.

## Required N, and whether these corpora supply it

For a 2.8pp absolute effect near a 74% base rate, two-sided alpha = 0.05, power
= 0.80, under the two-proportion normal approximation:

```
n = (z_a * sqrt(2 * p_bar * q_bar) + z_b * sqrt(p1*q1 + p2*q2))^2 / (p1 - p2)^2
  with p1 = 0.740, p2 = 0.768, z_a = 1.95996, z_b = 0.84162
```

That is **3,713 labelled queries per arm**, 7,426 in total.
<!-- derived: scripts/budget_discriminability_census.py#required_n_per_arm = 3713 -->

These corpora do not supply it, and the shortfall is not close. They hold 23
labelled queries in total before any exclusion, which is short of one arm's
requirement by more than two orders of magnitude. Hold that against whatever N
the census establishes before you read any EC value.

Note what the comparison is worth when the antecedent in
[Criteria that cannot fire on these corpora](#criteria-that-cannot-fire-on-these-corpora)
holds. If no cap binds anywhere in the grid, the ceiling is zero rather than
small, and no N closes that: the power calculation is then beside the point,
and reporting the run as underpowered would understate it.

## What a verdict can and cannot license

### Can

* Report the finding wording, with the N the census establishes.
* Declare the question unanswerable on these corpora, and state the reopen
  condition: a corpus whose labelled queries join gold by belief id and whose
  per-query pools price above the smallest shipped budget of 300 tokens.
* Keep every constant where it is.

### Cannot

* **Say the budgets are correct.** The census cannot reach that claim from any
  value of EC.
* **License a raise.** A ceiling statistic says the two arms could return
  different text, never which text is better, and a raise moves the pack in
  both directions — item 1 above.
* **License a cut.** The third cut condition needs a latency arm that does not
  exist.
* **Generalize past the configuration in
  [The instrument's disclosed limits](#the-instruments-disclosed-limits).** The
  verdict is scoped to an edgeless store, BFS off, the structural lane
  unexercised, and the block ceiling never applied. It says nothing about a
  production store, which has edges.
* **Treat the containment zero or the violation zero as a property of the
  instrument.** Both are properties of these corpora; their power comes from
  the K2 mutation arms.
* **Settle K1, K3, or the latency arm.** All three stay gated on a further
  ruling.

## How to run it

```
uv run python scripts/budget_discriminability_census.py --dry-run      # the plan, no store opened
uv run python scripts/budget_discriminability_census.py                # the full report
uv run python scripts/budget_discriminability_census.py --emit-figures # the published keys
uv run python scripts/budget_discriminability_census.py --check        # determinism
uv run pytest tests/test_budget_census.py -q                           # the bound, by mutation
```

Every mode exits non-zero on failure, and a bound violation is a failure. The
read-out lands in `docs/design/budget_discriminability_k0_result.md`.
