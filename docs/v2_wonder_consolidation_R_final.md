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
