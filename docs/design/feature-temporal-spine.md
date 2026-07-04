# Temporal spine — ingest-time chronological edges + dedicated retrieval lane (#1064)

Status: **landed default-off** (writer + backfill + lane). The default-ON
flip is gated on the pre-registered criteria in § Flip gate below.

## Mechanism

Three components, all deterministic and embeddings-free (#605):

1. **Spine writer** (`aelfrice.temporal_spine.write_temporal_spine`,
   wired into ingest behind `[ingest] write_temporal_spine` /
   `AELFRICE_TEMPORAL_SPINE_WRITE`, default off). After each belief
   insert, link to the previous belief in the same `session_id`
   (order `created_at`, tie-break insertion order) with
   `TEMPORAL_NEXT`, src = successor, dst = predecessor (the
   `models.py` semantics #386 settled), weight 0.8. One edge per
   belief (~1.0 edges/belief measured), O(1) per insert.

2. **Backfill** (`aelf spine backfill`, hidden subcommand). Idempotent
   per-session chain build over an existing store (insert-if-absent per
   `(src, dst, type)`), `--dry-run` counting mode, plus an `aelf
   doctor` row (spine present/absent, edge count). Existing stores
   predate the writer; the migration story cannot be "re-ingest
   everything". Pinned by test: backfill output == writer output on the
   same corpus.

3. **Retrieval lane** (`use_temporal_spine` /
   `AELFRICE_TEMPORAL_SPINE`, default off). Additive candidate source
   after L1: traverses `TEMPORAL_NEXT` from the top-5 packed L1 seeds,
   both directions, depth 1, node budget 32
   (`temporal_spine_budget` / `AELFRICE_TEMPORAL_SPINE_BUDGET`).
   Appended after L1 candidates — never displaces them pre-packing.
   No-op guard via `count_edges_by_type()`: zero spine edges → empty
   lane at ~zero cost, byte-identical output for spineless stores.
   Telemetry: `LaneTelemetry.temporal_spine` (packed survivors) +
   `temporal_spine_candidates` (pre-pack discoveries); the delta is the
   trim loss the G2 gate asks about. Soft-deleted beliefs
   (`valid_to` set) are skip-but-continue at traversal time so chain
   integrity survives GC; spine hits do **not** seed BFS (unmeasured
   surface — the confirmatory evidence ran depth-1 append-after-L1
   with BFS untouched).

## Why this works — and why it's a lane, not a gate change

Metric: gold-set coverage = |gold ∩ retrieved| / |gold| per question
(the lens that matters for aggregation/temporal questions with diffuse
gold; single-hit recall stays saturated and misses this). Prior rounds
established that ~84% of the gold missing at wide config shares **zero**
salient terms with the question — unreachable by any lexical means. The
spine reaches that gold through chronological adjacency to beliefs that
*do* match.

- **Dev (LongMemEval, 475 questions, l1=200/budget=8000):** overall
  coverage 0.531 → 0.658 (+12.7pp); temporal-reasoning +13.6pp;
  multi-session +11.1pp. A shuffled control — identical edge count,
  endpoints permuted — gains ~nothing (311× ratio). Deterministic rerun
  byte-identical.
- **Confirmatory (LoCoMo, 1,979 QA with evidence-belief gold sets, one
  shot, criteria pre-registered):** overall **+14.6pp** (0.460 → 0.606);
  temporal +17.2pp; multi-hop +10.4pp; 10× the shuffled control;
  all-evidence rate 0.133 → 0.260. Out-of-sample gain exceeded dev.
  Expansion-node budget curve is monotone (0.605 / 0.631 / 0.659 at
  32/64/128, ~+2.5pp per doubling, no plateau) — the effect is
  budget-limited, not substrate-limited.

The permanent ablation for this lane is
`benchmarks/temporal_spine_ablation.py` (gold-evidence coverage on
LoCoMo; arms: baseline / +spine / seeded shuffled-control).

Why a lane and not a #741 gate exception: #977's keep-off verdict is
correct *for generic BFS* and stays untouched; #741 explicitly
out-scoped per-edge-type gating; and the default BFS knobs structurally
suppress temporal traversal (`BFS_EDGE_WEIGHTS[TEMPORAL_NEXT] = 0.25` ×
`min_path_score = 0.10` prunes ≥2-hop chains, and temporal ranks last
per hop). The measured gains happened under depth-1-only traversal —
they are a floor.

Distinct from the #998 A4 ratified decline: A4 declined token-Jaccard
*co-occurrence* edges fed through the #981 HRR-expand lane ("density is
a liability"). Different edge class (similarity vs chronology),
different consumer. The shuffled control is the direct answer to the
density concern: identical density with scrambled endpoints recovers
+1.5pp vs the spine's +14.6pp — the value is the chronology, not the
density.

## Flip gate — pre-registered default-ON criteria

Default-off is the **landing posture, not the end state**. When all
pass, the next release flips both flags (writer + lane) default-ON in
one release, with the backfill path included for existing stores:

- **G1 — confirmatory evidence:** DONE (above; recorded in #1064).
- **G2 — production operating point:** coverage delta + top-rank
  invariance at the production hook budget (1500 tokens) on bench pools
  and a shadow eval on a real backfilled store (aggregate-only). Pass:
  ≥ +3pp coverage at production budget, no top-rank regression.
  (Dev/confirmatory ran at budget 8000; #1045/#1062 established the
  trim binds in ≥21% of real injections — this gate answers "does it
  survive the trim." Read `LaneTelemetry.temporal_spine_candidates −
  temporal_spine` for the per-call trim loss.)
  - **Bench-pool half: DONE.** `temporal_spine_ablation.py
    --budget 1500 --l1-limit 50 --rank-invariance` on LoCoMo10 (1,986
    questions): coverage **+19.45pp** (0.346 → 0.540; seeded shuffled
    control +0.47pp) — survives the trim, gain *exceeds* the wide-budget
    dev figure. Top-rank invariance: **1,986/1,986 questions
    core-prefix invariant, 0 top-rank displacements, 0 core-length
    mismatches** — the lane never displaces or reorders a
    `[locked, l25, l1, hrr]` belief; it only appends its own hits below
    the core (17,110 spine beliefs added across all questions, every
    one below the core). The `--rank-invariance` pass reads
    `last_lane_telemetry()` to locate the core boundary exactly, so the
    invariant is checked against the real lane structure, not an
    inferred one.
  - **Shadow-eval half: OPEN.** Still needs the aggregate-only run on a
    real hook-ingested backfilled store (see open question 1 —
    chain-length distribution under production `session_id` semantics).
- **G3 — latency delta (#739-style):** with spine present at ≥10k
  beliefs: p50 Δ ≤ 5 ms, p95 Δ ≤ 50 ms vs lane-off. (Generic BFS
  measured +1.0 ms p50 / +35.6 ms p95; this lane is narrower.)
  - **DONE.** `temporal_spine_latency.py` on a 10,000-belief / 200-session
    store carrying a real 9,800-edge spine, at the production operating
    point (budget 1500 / l1-limit 50), paired lane-off vs lane-on: **Δp50
    ≤ +0.8 ms, Δp95 +25–28 ms** (3 runs: +0.78/+0.04/+0.83 p50, worst
    +28.4 p95), tail ratio ~2.1× — inside the +5 ms / +50 ms / 10× band.
    The lane fires on 30/30 queries (109 candidates → 82 packed
    survivors), byte-identical across runs, so the delta is real work,
    not a vacuous null. Absolute latencies are machine/load-dependent
    (dev-run baseline p50 ~64 ms); the gate is a same-corpus *delta*, as
    reframed for #739 in PR #754. Pure-logic + lane-fires unit tests run
    in the CI matrix; the timed bench is run-on-demand (no wall-clock
    assertion in pytest — latency gates flake on shared runners).
- **G4 — migration:** backfill shipped + doctor row (DONE in the
  landing PRs); the flip release decides auto vs prompted backfill.
- **G5 — determinism/repro:** two-build byte-identity of the spine
  table on a fixed corpus; ablation bench green in CI.

## Open questions (tracked for the flip review)

1. Production `session_id` semantics: hook-ingested beliefs chain
   within a host session — chains will be long and heterogeneous. The
   first G2 shadow eval should watch chain-length distribution.
2. ~~Soft-deleted/superseded beliefs~~ — resolved skip-but-continue at
   traversal time (implemented; predecessor selection at write time
   also keeps GC'd beliefs eligible so chains never sever).
3. Federation: spine edges are local-store only; foreign-scope beliefs
   are excluded from chains (the writer runs on local inserts only).
4. ~~Ablation bench upstream~~ — done:
   `benchmarks/temporal_spine_ablation.py`.

## Historical note

The predecessor codebase added an equivalent writer (`link_temporal`)
five days *after* its last quantified benchmark run — this mechanism had
never been benchmarked before the #1064 campaign, on either codebase.
