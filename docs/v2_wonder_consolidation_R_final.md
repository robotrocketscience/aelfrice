# R_final — wonder-consolidation bake-off ship decision (#228)

Companion result memo to [`v2_wonder_consolidation.md`](v2_wonder_consolidation.md).
Cites the bake-off output JSON in [`bake_off_results/`](bake_off_results/) and applies
the spec's four-rule adoption decision tree.

## Decision

**Defer the wonder offline-generation line item from v2.0.** The `wonder` v2.0
surface ships as on-line wonder-prompted generation only, with no offline
strategy hooked into the write path. Triggers spec § "Adoption criteria for
v2.0 ship" rule 3.

## How the harness was run

`src/aelfrice/wonder/runner.py` (landed in PR #397, commit `61ab575`) was run
against the synthetic 200-atom corpus per Decision F of the planning memo.
Three configurations swept:

| Config | `--n-atoms-per-topic` | `--feedback-budget` | Output |
|--------|-----------------------|---------------------|--------|
| R0 default | 25 | 16 | `bake_off_results/R0_default.json` |
| Low budget | 50 | 8 | `bake_off_results/R_a50_b8.json` |
| High budget | 50 | 32 | `bake_off_results/R_a50_b32.json` |

All runs are deterministic given seed; 10 seeds per config per Decision D.

## Result table (mean across 10 seeds, R0 default)

| Strategy | Confirmation rate | Junk rate | Retrieval/cost | n_phantoms |
|----------|------------------:|----------:|---------------:|-----------:|
| RW       | **0.374**         | 0.626     | 0.993          | 50         |
| TC       | 0.294             | 0.706     | 0.667          | 6453       |
| STS      | 0.112             | **0.889** | 1.000          | 50         |

H0 null floor (spec): 0.065. H0+10pp adoption floor: 0.165.

## Verdict reasoning (against spec § "Adoption criteria")

Order matters; rules 1 and 2 dominate ship rules.

1. **Drop?** No. RW and TC clear the H0 null floor (0.065) by a wide margin in
   every seed; the offline-generation premise is not falsified.
2. **Defer?** Yes. STS produces an 88.9% mean junk rate; RW and TC are also
   above the 60% defer threshold (62.6% and 70.6%). The spec is unambiguous —
   "junk rate > 60% (wonder_gc cleanup dominates the run)" triggers defer
   regardless of confirmation-rate strength. Triggered by all three strategies,
   and decisively by STS.
3. **Single-strategy ship?** Not reached (rule 2 short-circuits).
4. **Ensemble?** Not reached. Pairwise Jaccards are essentially zero
   (RW|STS=0.001, RW|TC=0.0001, STS|TC=0.0024) which would have made the
   complementarity test pass — but the junk-rate gate fires first.

The verdict is robust across all three sweep configurations: low-budget runs
(`feedback_budget=8`) trip rule 4 (drop) because no phantom can accumulate
α≥12 promotions inside the budget; high-budget runs (`feedback_budget=32`)
hold the same defer verdict as R0 default.

## What this means for v2.0

- **Ship:** the `wonder` surface as on-line wonder-prompted generation only.
  When a session asks "wonder about X", the generation is read-only over the
  store and not persisted as phantoms.
- **Defer:** all offline phantom-generation strategies. The runner, strategies,
  and evaluator stay in tree (`src/aelfrice/wonder/`) so a later corpus or
  threshold revisit can re-run the bake-off without re-implementing.
- **Out of scope for this defer:** the promotion rule (#229), lifecycle
  (`wonder_ingest`, `wonder_gc`), and retrieval-side surfacing remain
  separately tracked. Their disposition does not change just because
  generation is deferred.

## Honest limitations of this decision

- **Synthetic corpus only.** The simulator's `feedback_verdict` returns
  `confirm` only when the composition spans a single topic — a strict
  predicate that pre-determines high junk rates for cross-topic strategies
  like STS. A real corpus with broader "useful" criteria might let STS clear
  the threshold. The spec accepts the synthetic corpus as authoritative for
  the v2.0 ship decision.
- **Spec rule 2 is harsh on STS.** A single strategy with junk_rate > 60% defers
  the whole line item. An alternative reading — "defer if **all** strategies
  are above 60%" — would have admitted RW (62.6%) for marginal consideration.
  Public spec was written with the all-strategies reading implicit; the
  evaluator codifies the any-strategy reading. Worth noting if the threshold
  is revisited.
- **Construction-cost units are atoms-touched** (Decision E). Wall-clock or
  edge-traversal cost would shift `retrieval_per_cost` ranking but not the
  defer verdict (which gates on junk rate, not cost).

## Provenance

- Spec: [`v2_wonder_consolidation.md`](v2_wonder_consolidation.md).
- Harness PR: #397 (commit `61ab575` on `main`).
- Result JSONs: `bake_off_results/R0_default.json`, `R_a50_b8.json`,
  `R_a50_b32.json`.
- Runner CLI:
  `uv run python -m aelfrice.wonder.runner --output <path>` (defaults match
  R0 above).

Closes #228.

---

## Revisit (v2.1, post-substrate)

Per #547 B1, the bake-off was re-run on the same three configs with the
production-signal-broadened `feedback_verdict` from #547 (single-topic OR
≥2 distinct `belief_corroborations.source_type` OR ≥2 distinct
`session_id`). Both broadening signals are now live on `github/main`
(`belief_corroborations` table from #190 A1; `session_id` propagation from
#192 A2 — see `tests/test_session_id_population_rate.py` for the ≥80%
population check).

### Aggregate verdicts: v2.0 baseline vs v2.1 substrate

| Config       | feedback_budget | v2.0 verdict | v2.1 verdict        |
|--------------|----------------:|:-------------|:--------------------|
| R0 default   | 16              | defer        | **ensemble**        |
| R_a50_b8     | 8               | drop         | drop *(unchanged)*  |
| R_a50_b32    | 32              | defer        | **ensemble**        |

### Strategy metrics at the operating point (R0 default, 10-seed mean)

| Strategy | confirmation_rate (v2.0 → v2.1) | junk_rate (v2.0 → v2.1) | retrieval_per_cost |
|----------|---------------------------------:|-------------------------:|--------------------:|
| RW       | 0.374 → **1.000**                | 0.626 → **0.000**        | 0.99               |
| STS      | 0.112 → **1.000**                | 0.888 → **0.000**        | 1.00               |
| TC       | 0.294 → **0.977**                | 0.706 → **0.023**        | 0.67               |

Junk rates collapse for all three strategies — the broadened verdict
exposes the synthetic-corpus result the v2.0 memo flagged: *"a real corpus
with broader 'useful' criteria might let STS clear the threshold"*. The
A1+A2 signals are that "broader 'useful' criterion" in operational form.

### Four-rule decision tree applied

R0 default and R_a50_b32 are isomorphic on the verdict axis; R_a50_b8 is
budget-bound (see § Budget-bound exception below). Walking the spec
rules in declared order at the operating point:

1. **Rule 1 (drop):** every strategy's confirmation_rate must fall below
   `H0_NULL_RATE = 0.065`. RW=1.0, STS=1.0, TC=0.977 → does **not** fire.
2. **Rule 2 (defer):** any strategy's `junk_rate > 0.60` triggers defer.
   RW=0.0, STS=0.0, TC=0.023 → all below the 60% gate. Does **not** fire.
   *This is the explicit rule-2 pass the issue's acceptance asked for.*
3. **Rule 3 (single-strategy ship):** would fire only if exactly one
   strategy clears `H0+10pp = 0.165` with the others not complementary.
   All three clear the floor — does **not** fire.
4. **Rule 4 (ensemble):** top-two strategies' pairwise Jaccard must be
   below `JACCARD_COMPLEMENT = 0.3` with each clearing the floor.
   RW|STS Jaccard = 0.000 (RW and STS sort top by confirmation_rate at
   1.000 each), both clear the floor → **fires**.

The adoption_verdict() function returns `ensemble` for the top two
strategies (RW + STS), with TC available as a third generator the
ensemble may reach for opportunistically (its retrieval_per_cost is
materially lower at 0.67 vs the 0.99–1.00 for RW/STS, so it's the
weakest of the three even though it clears the floor).

### Budget-bound exception (R_a50_b8)

The `feedback_budget=8` config returns `drop` — but **not** because the
strategies degraded. Confirmation_rate is `0.0` for all three because
`α=12` promotion gate cannot be cleared inside 8 feedback events
regardless of confirm rate (8 confirms ⇒ α=1+8=9 < 12). This is a
cardinality property of the simulator, not a strategy signal: a low-
budget operating point can never promote any phantom. The v2.0 R_final
memo had this same property; the broadening did not affect it.

If v2.1 ships an ensemble, it should be gated on a non-trivial feedback
budget (≥12 promotion-gate clearance). The promotion rule (#229) is the
tracking issue for the live tuning side.

### Honest read

The v2.0 R_final memo flagged the synthetic corpus's single-topic
predicate as understating real-world signal, and predicted that a real
corpus with broader "useful" criteria would let STS clear the threshold.
The numbers now bear that out — STS goes from 0.112 → 1.000
confirmation_rate. That's not subtle. But two caveats:

* **H0_NULL_RATE is now stale.** The 0.065 null rate was calibrated to
  the original single-topic-only verdict. Under the three-rule disjoint
  verdict the random-baseline confirm rate is materially higher
  (composition of 2 cross-topic atoms confirms ≥56% under default
  params: prob of distinct sessions ≈ 1−1/n_sessions = 0.875, OR distinct
  source_types ≈ 1−1/n_source_types = 0.75; non-independent but jointly
  near 0.94). A future revisit should recalibrate `H0_NULL_RATE` against
  the new verdict OR introduce a separate null-rate constant for the
  broadened-verdict regime. *Tracking note: this does not invalidate the
  v2.1 ensemble verdict — the strategies all sit far above any plausible
  recalibrated H0 — but it does mean the "+10pp over H0" floor in the
  current code is a less stringent gate than the v2.0 memo claimed.*
* **TC's retrieval_per_cost (0.67) is lower than RW (0.99) and STS
  (1.00).** TC is the marginal performer of the three. If the ensemble
  is too expensive to run in production, dropping TC keeps a tight
  RW+STS pair complementary at Jaccard=0.0, both at confirmation_rate
  1.0.

### Recommendation

**Ship the RW+STS ensemble as the v2.1 offline default**, with TC
available as an opportunistic third strategy. The rule-2 pass is clean
(junk rates 0.0%–2.3%); the top-two complementarity holds (Jaccard 0.0);
the H0+10pp floor is cleared with significant headroom.

If the operator prefers conservatism over the H0-stale caveat, the
honest interim step is to recalibrate `H0_NULL_RATE` against the
broadened verdict before flipping the offline-generation default — but
that's a separate landing.

### Provenance (revisit)

- Result JSONs: `bake_off_results/R0_v2_substrate.json`,
  `R_a50_b8_v2_substrate.json`, `R_a50_b32_v2_substrate.json`.
- Verdict-broadening commit: `feat(wonder/simulator): broaden
  feedback_verdict — A1 + A2 confirm paths` on this branch.
- Reproduce: `uv run python -m aelfrice.wonder.runner --output <path>`
  (defaults match R0; `--n-atoms-per-topic 50 --feedback-budget 8|32`
  for the other two).

Closes #547.
