# Budget discriminability decision rule (#1546)

> **Voided on 2026-09-23. The live pre-registration is
> `docs/design/budget_discriminability_k0_preregistration.md`.**
>
> The operator voided this document because it described the instrument
> wrongly in four places — it called the cluster packer inert, treated probe
> containment as a theorem rather than a property of these corpora, inherited a
> containment mutation guard that had no power on three of six lanes, and
> attributed BFS's absence to the missing edges rather than to a default-off
> flag. Read the replacement instead. It carries every criterion below forward
> unchanged and adds the disclosures this document lacked.
>
> This document is retained because the replacement quotes it and because
> `scripts/budget_discriminability_census.py` still names it as
> `report()['decision_rule']`. Nothing below is amended after the void: an
> amendment here would be a criterion change on a voided rule, which decides
> nothing.

This document is the pre-registration for the injection-budget re-tune
([#1546](https://github.com/robotrocketscience/aelfrice/issues/1546)). You are
reading the rule that decides the outcome. It is written and committed **before
any number from the census exists**, so the budgets cannot be fitted to
whichever arm wins.

The operator ruling of 2026-09-18 orders the kill experiment first: build the
discriminability census, run K0, and move no constant. K1 (LongMemEval-S), K3
(the A/A noise replicate), and the latency arm are gated on K0 and on a further
ruling. They are out of scope here.

If you change anything below after a census number has been seen, the run is
void and you start again from a fresh pre-registration.

## Amendments

### Amendment 1, 2026-09-17: the monotonicity claim was false

This rule asserted that admission is monotone in budget. It is not. The
sentences that said so are corrected in place below, and the originals are
quoted here so the amendment is auditable rather than a quiet rewrite:

> Admission is monotone in budget: raising a budget can only admit beliefs,
> never evict them.

> Admission is monotone across the grid: raising a budget never removes a
> belief that a lower budget admitted.

What the code does instead: `clustering.pack_with_clusters` is a
skip-and-continue greedy fill. Stage 1 abandons on the first representative it
cannot afford, and stage 2 then walks the rest of the pool, so the budget
*selects* rather than truncates. Through `retrieval.retrieve()` with default
resolvers, on a two-belief store priced at 63 and 19 tokens, raising
`token_budget` from 62 to 63 evicts the belief the lower budget admitted:

```
  62 -> ['lean']
  63 -> ['fat']
```

`tests/test_budget_census.py::test_raising_a_budget_can_evict_a_belief_a_lower_budget_admitted`
pins that, and fails against a monotone prefix packer.

What the correction changes, and what it leaves alone:

* **The operative bound is untouched.** It is conditional — a pool priced at or
  below the smaller of two budgets renders an identical block — and it
  quantifies only over budgets at which no cap binds. Non-monotone selection
  happens only where a cap binds, so the two never meet.
* **No decision criterion moves.** EC, N, the grey band, the required N, the
  three cut conditions, the forbidden fallbacks, and the finding wording are
  all unchanged.
* **The verdict on a raise stands and its reason is replaced.** A raise is
  still not licensed here, now because a raise can evict as well as admit: even
  "more beliefs are admitted" is not a correct description of the mechanism,
  let alone evidence that the result is better.
* **Whether a mechanism correction voids a pre-registration is the operator's
  call, not the correcting author's.** If it does, the replacement is a fresh
  pre-registration and a re-run. The census's N and EC do not depend on the
  corrected sentence, so a re-run returns the same numbers.

### Amendment 2, 2026-09-17: deviations this rule did not disclose

* The grid holds `hook.HOOK_BLOCK_TOKEN_CEILING` fixed, which the table below
  calls invalid, and the census never applies `hook.enforce_block_ceiling` at
  all: it measures `retrieve()` output rather than a rendered hook block. The
  ceiling is 6000 and the dearest pool the census prices is 132 tokens, so the
  ceiling cannot bind on these corpora. The deviation is disclosed, not
  repaired.
* The census seeds beliefs and no edges, so the temporal spine, BFS expansion,
  and cluster structure are inert. Those are the sources where a budget could
  change candidacy rather than count, so EC is measured on a retrieval
  configuration narrower than production.
* `retrieval._route_structural_query` is default-on and unexercised: it fires
  only on a `<KIND>:<target_id>` marker query, and neither corpus holds one. It
  also prices with `retrieval._belief_tokens` and ignores `belief_cost_fn`.

The census prints all three under "what this run does not exercise".

## The constants under adjudication

Each value is read at its definition site by symbol, not copied from a prior
summary. The values below are what the tree holds at the commit that introduces
this document.

| Symbol | Value | What it bounds |
| --- | --- | --- |
| `hook.DEFAULT_HOOK_TOKEN_BUDGET` | 1500 | the `UserPromptSubmit` belief block |
| `hook.DEFAULT_SESSION_START_CORE_TOKEN_BUDGET` | 1500 | the SessionStart `<core>` section |
| `hook_search_tool.INJECTED_TOKEN_BUDGET` | 600 | the `Grep`/`Glob` PreToolUse lane |
| `hook_search_tool.BASH_INJECTED_TOKEN_BUDGET` | 300 | the `Bash` PreToolUse lane |
| `hook_agent_context.INJECTED_TOKEN_BUDGET` | 600 | the Agent/Task tool-dispatch lane |
| `retrieval.DEFAULT_TOKEN_BUDGET` | 2400 | every `retrieve()` caller that passes none |
| `rebuild_log.DEFAULT_REBUILDER_TOKEN_BUDGET` | 4000 | the v1.4 rebuild block |

Two adjacent symbols move with them, and a grid that holds either fixed is
invalid:

| Symbol | Value | Why it moves |
| --- | --- | --- |
| `hook.HOOK_BLOCK_TOKEN_CEILING` | 6000 | defined as `4 * DEFAULT_HOOK_TOKEN_BUDGET`, so it is not independent |
| `retrieval.DEFAULT_L25_TOKEN_SUBBUDGET` | 400 | the cross-lane L2.5 sub-cap, and #1546's own prerequisite 2 |

`retrieval.RELEVANCE_BUDGET_FLOOR_FRACTION` is 0.5 and is not under
adjudication, but it sits between a budget and the sub-cap: the effective L2.5
sub-budget is `min(DEFAULT_L25_TOKEN_SUBBUDGET, max(int(B * 0.5), B -
locked_cost))`. Any census that varies `B` without varying the sub-cap measures
the floor, not the budget.

### One constant does not move regardless of outcome

`rebuild_log.DEFAULT_REBUILDER_TOKEN_BUDGET` is frozen by this rule. It is
reachable as `[rebuilder] token_budget` in a user's `.aelfrice.toml`, so
changing the default silently reinterprets a value a user has already written
against the old meaning. Moving it is a separate decision with its own
migration question, not a by-product of a re-tune.

## The primary statistic

**EC = D / N.**

* `N` is the count of non-degenerate labelled queries in the corpora under
  test. A query whose gold set is its entire candidate pool is **degenerate**: a
  budget can only truncate, so no budget can change which gold beliefs are
  present when every candidate is gold. Degenerate queries are excluded from
  `N`, counted, and reported.
* `D` is the count of those queries whose **gold-belief footprint in the
  rendered lane block** differs between the two arms. The footprint is the
  ordered sequence of rendered lines for gold beliefs, taken from the lane's own
  renderer — not the belief content, and not the whole block, because a
  per-line character cap can change what a gold belief says while leaving its
  id in place.

EC is an exact **ceiling** on effect, and nothing else. It is a change
statistic in the sense `benchmarks/README.md` defines: it says the two arms
could return different text, and says nothing about which arm is right. It must
never be reported as a quality result, an uplift, or a regression.

### The exact clause

**EC = 0 means zero discordant pairs are possible, not merely unobserved.** If
every candidate pool prices below the smaller of the two budgets, both arms
render byte-identical blocks, the model receives identical input, and every
downstream metric is identical by construction. No sample size over these
corpora detects any effect, because there is no effect to detect at any sample
size. Report that as an **impossibility result**. Reporting it as an
underpowered null is wrong: an underpowered null says "we could not see it", and
this says "it is not there to see".

## The grey band

`NF = max(1.96 * sqrt(0.25 / N), 9.5pp, A/A_band)`.

* The first term is the two-sided normal half-width for a proportion at its
  worst-case variance, at the `N` the census establishes.
* The 9.5pp term is the #1255 proposal-17 inter-grader spread. It is a
  historical prior carried forward, not an estimate produced here. An effect
  smaller than the disagreement between two human graders on the same answers is
  not a result this repo acts on.
* `A/A_band` is the K3 replicate's observed spread. K3 is built as of
  [#1546](https://github.com/robotrocketscience/aelfrice/issues/1546) and
  measures 0.0pp, so the band is the max of all three terms and the run is
  labelled as carrying an A/A term. This document is superseded — see the
  banner above and `budget_discriminability_k0_preregistration.md`, which
  carries the current statement.

The band may not be narrowed after a number is seen. If K3 later widens it, the
wider band applies retroactively to any verdict that used this rule.

## What this instrument may and may not license

### A raise is never licensed by this instrument

A ceiling statistic cannot answer "should we raise it". It says only that the
two arms could return different text, never which text is better. Raising a
budget moves the pack in both directions — the fill skips what it cannot afford
and keeps going, so a raise admits some beliefs and evicts others (Amendment 1)
— and neither the admissions nor the evictions are evidence that the result
improves.

If `EC > NF` and the effect points up, the published verdict is exactly this
sentence and no other:

> A raise is not excluded; referred to an answer-quality gate that does not
> exist.

### A cut is the only direction this instrument may license alone

A cut arm is licensed only when all three hold:

1. its EC is within `NF`;
2. its `binds_on` is not `pool` — a cut whose binding cap is the pool is a cut
   that changed nothing, and licensing it would be licensing a no-op as if it
   were a finding;
3. latency does not worsen.

Item 3 needs the latency arm, which is out of scope here. K0 alone therefore
cannot license a cut either. K0 can only tell you whether the question is
answerable at all.

### The forbidden fallbacks

None of the following may be substituted as the quality verdict, in the
published finding or in a follow-up summary:

* injected bytes or any byte count;
* item counts, or "N of M queries changed";
* top-k movement rate;
* Jaccard overlap at k;
* Kendall tau.

This repo has already retracted a published table for treating a change
statistic as quality evidence — see `benchmarks/README.md`, "What these
harnesses measure, and what they do not", and commit `848dbf83`.

## Required N

For a 2.8pp absolute effect near a 74% base rate, two-sided alpha = 0.05, power
= 0.80, using the two-proportion normal approximation:

```
n = (z_a * sqrt(2 * p_bar * q_bar) + z_b * sqrt(p1*q1 + p2*q2))^2 / (p1 - p2)^2
  with p1 = 0.740, p2 = 0.768, z_a = 1.95996, z_b = 0.84162
```

That is **3,713 labelled queries per arm**, 7,426 in total. Hold that number
next to whatever `N` the census establishes over the public corpora before
reading any EC value.

## The finding wording

When the run lands, the finding is worded as:

> unmeasurable at N=<n>; reopen at belief-id-joined gold with per-query pools
> exceeding the smallest shipped budget

Never "the budgets are correct". The census cannot say that. It can say only
that these corpora cannot tell the arms apart.

## The premise this rests on, and how it is checked

A budget only ever truncates a pack. The claim that follows — that a pool
pricing below `min(B1, B2)` forces byte-identical output — is load-bearing, so
it is demonstrated rather than read off the source:

* `retrieval._l1_hits` takes no budget parameter, so candidate generation does
  not vary with the budget.
* `retrieval._l25_hits` **does** receive a budget-derived cap
  (`effective_l25_subbudget`), so a prior claim that the budget reaches neither
  lane is only half right. The L2.5 trim is a tail truncation, but the census
  varies the sub-cap because of it.
* `clustering.pack_with_clusters` skips an over-budget belief and keeps
  filling, so a budget at or above total pool cost returns the whole pool.
* Admission is **not** monotone across the grid: raising a budget can remove a
  belief that a lower budget admitted (Amendment 1). The bound survives because
  it is conditional and quantifies only over budgets at which no cap binds. No
  argument in this rule may reason from monotonicity.

`tests/test_budget_census.py` holds each of these, and holds the census itself
against three deliberately budget-sensitive fake packers: one that truncates by
item count, one that returns less at a higher budget, and one whose budget
moves only a non-gold belief. Until each fake makes the census report a
violation and the shipped code reports none, the bound is **unverified** and
every number the census prints inherits that caveat.
