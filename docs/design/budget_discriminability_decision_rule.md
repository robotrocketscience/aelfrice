# Budget discriminability decision rule (#1546)

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
| `hook_agent_context.INJECTED_TOKEN_BUDGET` | 600 | the subagent-dispatch lane |
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
* `A/A_band` is the K3 replicate's observed spread. K3 is not built yet, so the
  band for K0 is the max of the first two terms, and the run is labelled as
  carrying no A/A term.

The band may not be narrowed after a number is seen. If K3 later widens it, the
wider band applies retroactively to any verdict that used this rule.

## What this instrument may and may not license

### A raise is never licensed by this instrument

Admission is monotone in budget: raising a budget can only admit beliefs, never
evict them. A ceiling statistic therefore answers "should we raise it" before
the run starts — the answer is always "more beliefs are admitted", which is a
description of the mechanism and not evidence that the extra beliefs help.

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
  lane is only half right. The L2.5 trim is still a tail truncation, which keeps
  the monotonicity argument, but the census varies the sub-cap because of it.
* `clustering.pack_with_clusters` fills its second stage with `continue`, not
  `break`, so a budget at or above total pool cost returns the whole pool.
* Admission is monotone across the grid: raising a budget never removes a
  belief that a lower budget admitted.

`tests/test_budget_census.py` holds each of these, and holds the census itself
against a deliberately budget-sensitive fake packer. Until that mutation guard
fails on the fake and passes on the shipped code, the bound is **unverified**
and every number the census prints inherits that caveat.
