# K3 result: the A/A band is zero, and the instrument could have said otherwise (#1546)

This is the K3 read-out. The registration it is read against is
`docs/design/budget_discriminability_k0_preregistration.md`, which defines the
census noise floor as

```
NF = max(1.96 * sqrt(0.25 / N), 9.5pp, A/A_band)
```

and records that the third term is missing. K3 supplies it.

The producer is `scripts/budget_discriminability_aa_replicate.py`. It is
committed to the tree, and every figure below that carries a marker comes out
of it under `--emit-figures` (#1469).

Re-derive the figures on this page with:

```
uv run python scripts/budget_discriminability_aa_replicate.py
uv run python scripts/budget_discriminability_aa_replicate.py --emit-figures
uv run python scripts/budget_discriminability_aa_replicate.py --check
uv run pytest tests/test_budget_aa_replicate.py -q
```

## The finding

`A/A_band` is **0.0pp** across 9 replicates of the census, and the positive
control shows the same replicate reporting 66.6667pp against a retrieval path
that reads the one thing the perturbation moves.
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_band_pp = 0.0 -->
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_replicates = 9 -->

Read the zero as "the paths this census exercises do not read insertion order",
never as "the census has no instrument noise". The band is scoped to the
perturbation, the corpora, and the retrieval configuration named below.

`NF` is unchanged by this result. At N = 7 the binomial term is 37.0398pp and
the inter-grader prior is 9.5pp, so `max(37.0398, 9.5, 0.0)` is still
37.0398pp. K3 changes what the band is *known* to contain, not its width.

## What K3 measures, and why a re-run would not

The census is deterministic, and
`scripts/budget_discriminability_census.py --check` proves it on every run by
hashing two reports. An A/A replicate that re-runs the same inputs therefore
measures exactly zero by construction and says nothing about the instrument.

The pre-registered perturbation is **belief insertion order**. For each
replicate the producer re-ingests every labelled query's beliefs under a seeded
permutation, holding the corpus, the gold sets, the grid, the lane set, and
every belief's content, id, type, origin, posterior, and timestamp fixed. Only
the rowid that SQLite assigns moves. Rowid is a live tie-break key in this
repo — the temporal spine orders by rowid rather than by time — so an
instrument that reads it moves under this perturbation while nothing a
measurement may depend on has moved.

The producer does not reimplement EC. It hands each permuted population to
`budget_discriminability_census.report(queries=...)`, so the lanes, the grid,
the cost functions, `measure_query`, the exclusion rules, the block-identity
check, and the containment check are the census's own. A replicate that
reimplements the statistic measures its own reimplementation.

Replicate 0 is the census's committed order, unpermuted. Its EC row reproduces
the published K0 figures, which is what ties this report to the census it
claims to replicate. Replicates 1 through 8 are the seeded permutations.

## The band, per cell

A *cell* is one EC value the census publishes: a lane total, or a lane crossed
with a corpus. The producer tracks 18 of them, and the band of a cell is
`max(EC) - min(EC)` over the replicates.
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_cells = 18 -->

| Lane | Band over 9 replicates |
| --- | --- |
| `ups` | 0.0pp |
| `search_tool` | 0.0pp |
| `search_tool_bash` | 0.0pp |
| `agent_context` | 0.0pp |
| `retrieval_default` | 0.0pp |
| `rebuilder` | 0.0pp |
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_band_pp.ups = 0.0 -->
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_band_pp.search_tool = 0.0 -->
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_band_pp.search_tool_bash = 0.0 -->
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_band_pp.agent_context = 0.0 -->
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_band_pp.retrieval_default = 0.0 -->
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_band_pp.rebuilder = 0.0 -->

Each lane's two per-corpus cells band at 0.0pp as well; run the producer
without `--emit-figures` to see all 18 rows.

N holds at 7 on every replicate, and the band on N is therefore 0, so the EC
band compares values computed over one population rather than over several. A
moving N is reported as a violation rather than folded into the band, because
EC values with different denominators are not comparable.
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_n = 7 -->
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_n_band = 0 -->

Across the replicates, the census reports 0 bound violations.
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_violations = 0 -->

## The positive control

A zero from an instrument that cannot report anything else is a print
statement, not a measurement. `tests/test_budget_aa_replicate.py` drives the
replicate against a fake `retrieval.retrieve` that reads the store's `rowid`
column and drops the pool member the store saw last, on the grid cells whose
L2.5 sub-budget is 200. Which belief it drops is decided by the ingest order
and by nothing else, so whether that belief is gold varies from replicate to
replicate and EC moves with it.

Under that fake, over the 3 replicates the test runs (`TEST_SEEDS = 2`, the
committed order plus two permutations):

| Arm | `aa_band_pp` | Per-lane band | Order-sensitive grid arms |
| --- | --- | --- | --- |
| shipped code | 0.0pp | 0.0pp on all six | 0 of 2070 |
| rowid-reading fake | 66.6667pp | 57.1429pp on all six | 150 of 2070 |

The test asserts on the producer's own `aa_band_pp`, per lane, not on a spread
the test computes itself: what is under test is the replicate's ability to
report a spread. A band carried by one lane while five report zero would read
as covered while five lanes were never exercised.

Two source mutations red the control, so its power rests on the producer's code
and not on the fixture:

| Mutation | Result |
| --- | --- |
| `permuted_corpora` returns the committed order for every replicate | 4 tests fail, the control among them |
| `spread` returns `0.0` unconditionally | the control fails, alone |
| `arm_outputs` returns a count of 0 | the sweep test fails, alone |

One consequence is worth stating plainly, because it decides how the producer
is written. On these corpora every candidate pool prices below every budget in
the grid, so any discordant gold footprint is also a broken block-identity
bound, and the census reports violations under the fake. The producer therefore
computes the band whatever the violation list says. A band gated on a clean
violation list could only ever report zero here, which is the exact defect this
replicate exists to rule out. Violations still fail the run; they do not
silence the measurement.

## Why the band is zero

The band answers whether the statistic moves. The producer also asks the
sharper question one level down: does anything the statistic reads move at all?
It sweeps every lane, query, and grid cell, and compares the retrieved belief
list across replicates.

Nothing moves: of the 2070 grid arms the sweep examines, 0 return a different
belief list under a different insertion order.
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_arm_cells_examined = 2070 -->
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_arm_cells_order_sensitive = 0 -->

That distinguishes a real zero from a cancelling one: the band is not zero
because order-sensitive movements happened to average out, it is zero because
the inputs to EC are invariant under the perturbation. The mechanism is
readable in the source:

* The FTS search orders by `bm25(beliefs_fts), b.id` — see `_ORDER_BY_BM25` in
  `src/aelfrice/store.py`. The tie-break is the belief id, a content-derived
  key, not the rowid.
* The entity-index and reference-manifest queries order by `b.id ASC` and by
  `overlap DESC, be.belief_id ASC`, on the same content-derived key.
* The one lane known to order by rowid is the temporal spine, and it never runs
  here: `census._open_store` writes beliefs and no edges, so
  `store.has_edge_type(EDGE_TEMPORAL_NEXT)` is `False` on every store this
  census builds.

So the zero is a property of the census's edgeless `:memory:` stores together
with the id tie-break, and the third bullet is the one a future corpus can
close. Re-run this producer against any corpus whose stores carry edges before
you carry this band forward.

## The perturbation, checked rather than assumed

A perturbation that changed nothing would make the whole replicate vacuous, so
the producer reports how many distinct insertion orders each labelled query
actually received, and the number of queries that received only one is 0.
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_queries_with_one_insertion_order = 0 -->

`tests/test_budget_aa_replicate.py::test_the_permutation_actually_moves_the_rowids`
asserts the same thing where it counts, against a real store: every query's
`(rowid, id)` assignment differs across replicates, and every query keeps
exactly the belief ids it started with. A permutation that dropped or added a
belief would be a different corpus, not a different order.

## Seeds, and what the count buys

The default is 8 permuted replicates on top of the committed order, under the
pinned base seed 1546. Each query's permutation seed is a SHA-256 digest of the
base seed and the query's identity, so it is stable across processes,
platforms, and `PYTHONHASHSEED` values (#605).
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_base_seed = 1546 -->
<!-- derived: scripts/budget_discriminability_aa_replicate.py#aa_seeds = 8 -->

The count is chosen against the 300-second producer timeout that
`scripts/check_derived_figures.py` enforces. A replicate costs one
`census.report()`, measured at 0.85 seconds, plus its share of the
order-sensitivity sweep; 9 replicates run in 56 seconds and 17 in about 105.

The band is a sample range, so it is monotone non-decreasing in the replicate
count: raising the count can only widen it. A low count can understate the
band and can never flatter it. Confirm that at a wider count with
`--seeds 32`, which holds the band at 0.0pp and the order-sensitive arm count
at 0 of 2070.

## What this does and does not license

* **The A/A term exists now.** `NF` may be written with three terms rather than
  two for any verdict read against this configuration.
* **`NF` does not narrow.** The band is 0.0pp, so the binomial term and the
  inter-grader prior still decide the floor. No verdict already read against
  the two-term band moves.
* **The band does not generalize past this configuration.** It is scoped to
  edgeless `:memory:` stores, BFS off, the structural lane unexercised, and the
  block ceiling never applied — the same narrowings the K0 registration
  discloses. A production store has edges.
* **K3 licenses no constant to move.** It is a noise term, not a quality
  result, and the forbidden fallbacks in the registration apply to it
  unchanged.

## What the evidence does not determine

* **Whether insertion order matters on an edge-bearing store.** The temporal
  spine orders by rowid and never runs here, so this band cannot speak to it.
  `tests/test_budget_census.py::test_containment_is_a_property_of_this_corpus_and_not_a_theorem`
  already shows that one `TEMPORAL_NEXT` edge changes what the census's own
  arms return; nobody has run K3 against such a corpus.
* **Whether a different perturbation would show a wider band.** Insertion order
  is the perturbation this registration names, and the choice is fixed in
  advance precisely so a later author cannot pick the one that yields the
  narrowest band. Other perturbations — a different noise order in the fixture
  file, a different store backend — are unmeasured, not ruled out.
* **The true spread.** A sample range over 9 replicates is a lower bound on it.
  The band's zero says no replicate in this sample disagreed with another, not
  that no permutation could.
