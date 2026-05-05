# v2.0 evaluation corpus — schema (#307)

Six bench-gated v2.0 modules ship/no-ship on positive impact against a labeled
corpus. This directory holds that corpus. The bench-gate harness (#319) reads it
via `AELFRICE_CORPUS_ROOT`; the modules in #193, #197, #199, #201, #228, #229
are evaluated against it. (#288 is the **rebuilder**-precision harness — a
different consumer.)

## Mounting on the lab side

Corpus content lives in the private lab repo only. Public CI runs with the
env var unset; bench-gate tests skip cleanly. Lab runs:

```bash
export AELFRICE_CORPUS_ROOT="$HOME/projects/aelfrice-lab/tests/corpus/v2_0"
./scripts/run_bench_gate.sh
```

## Layout

```
tests/corpus/v2_0/
├── README.md                          (this file)
├── dedup/                             #197
│   └── *.jsonl
├── enforcement/                       #199
│   └── *.jsonl
├── contradiction/                     #201
│   └── *.jsonl
├── wonder_consolidation/              #228
│   └── *.jsonl
├── promotion_trigger/                 #229
│   └── *.jsonl
├── sentiment/                         #193
│   └── *.jsonl
├── directive_detection/               #374 (H1 split of #199)
│   └── *.jsonl
├── bfs_relates_to/                    #383 (Track A edge: RELATES_TO)
│   └── *.jsonl
├── derived_from_edge/                 #388 (Track A retroactive gate: DERIVED_FROM)
│   └── *.jsonl
├── temporal_next_edge/                #386 (Track A edge: TEMPORAL_NEXT)
│   └── *.jsonl
├── tests_edge/                        #384 (Track A edge: TESTS)
│   └── *.jsonl
├── retrieve_uplift/                   #154 (v1.7 default-on flip — per-flag NDCG@k)
│   └── *.jsonl
├── reasoning/                         #389 (Track B: aelf reason)
│   └── *.jsonl
└── wonder_online/                     #389 (Track B: aelf wonder)
    └── *.jsonl
```

One JSONL file per logical batch so each bench gate has independent fixtures
and `git diff` stays readable. Filenames are advisory; the harness globs
`*.jsonl` per module directory.

## Per-line shape

Every line is a JSON object. Field set by module. Common envelope fields are
required for **all** modules:

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable unique id within the module file. Conventional prefix `<module>-<short>-<NNN>`. |
| `provenance` | string | Where the example came from (transcript hash, doc ref, "synthetic-v0.X", etc.). Free text; required to be non-empty. |
| `labeller_note` | string | One-line rationale for the label. Required to be non-empty. Used for audit + future re-labelling. |
| `label` | string | Module-specific. Allowed values listed below. |
| `seed` | bool (optional) | `true` for examples committed only to anchor the schema. **Seed rows do NOT count toward the v0.1 ≥50/module target.** Default `false` if omitted. |

### Module-specific fields

| Module | Extra required fields | Allowed `label` values |
|---|---|---|
| `dedup` | `belief_a` (string), `belief_b` (string) | `duplicate`, `near-duplicate`, `distinct` |
| `enforcement` | `user_directive` (string), `agent_output` (string) | `compliant`, `violated`, `n/a` |
| `contradiction` | `belief_a` (string), `belief_b` (string) | `contradicts`, `refines`, `unrelated` |
| `wonder_consolidation` | `seed_belief` (string), `retrieved_neighbors` (list[string]) | `1`, `2`, `3`, `4`, `5` (phantom-quality rating) |
| `promotion_trigger` | `belief_sequence` (list[string]) | `should_promote`, `should_not` |
| `sentiment` | `user_message` (string) | `positive`, `negative`, `neutral` |
| `directive_detection` | `prompt` (string) | `directive`, `not_directive` |
| `bfs_relates_to` | `beliefs` (list[obj]), `edges` (list[obj]), `seed_ids` (list[string]), `expected_hit_ids` (list[string]), `k` (int) | `graded` |
| `implements_edge` | `beliefs` (list[obj]), `edges` (list[obj]), `seed_ids` (list[string]), `expected_hit_ids` (list[string]), `k` (int) | `graded` |
| `derived_from_edge` | `beliefs` (list[obj]), `edges` (list[obj]), `seed_ids` (list[string]), `expected_hit_ids` (list[string]), `k` (int) | `graded` |
| `temporal_next_edge` | `beliefs` (list[obj]), `edges` (list[obj]), `seed_ids` (list[string]), `expected_hit_ids` (list[string]), `k` (int) | `graded` |
| `tests_edge` | `beliefs` (list[obj]), `edges` (list[obj]), `seed_ids` (list[string]), `expected_hit_ids` (list[string]), `k` (int) | `graded` |
| `retrieve_uplift` | `query` (string), `beliefs` (list[obj]), `edges` (list[obj]), `expected_top_k` (list[string], **ordered**), `k` (int) | `graded` |
| `reasoning` | `query` (string), `beliefs` (list[obj]), `edges` (list[obj]), `expected_hit_ids` (list[string]), `baseline_search_only_top_k` (list[string]), `k` (int) | `graded` |
| `wonder_online` | `beliefs` (list[obj]), `edges` (list[obj]), `seed_id` (string), `expected_candidate_ids` (list[string]) | `graded` |

### `directive_detection` re-entry gate (#374)

Distinct from the other modules: this gate decides whether H1 of #199
unblocks for implementation, not whether a shipped module continues to
ship. Per `docs/v2_enforcement.md` § H1, H1 reopens for implementation
when **all three** are true on a labeled sample of ≥ 200 coding prompts:

- precision ≥ 0.80 (true-directive / predicted-directive)
- recall ≥ 0.60 (true-directive / actual-directive)
- sample published (this directory; synthetic-only per directory-of-origin
  rules — real coding-prompt transcripts stay lab-side)

If those numbers are not met, H1 stays deferred. The bench-gate test at
`tests/bench_gate/test_directive_detection.py` evaluates the candidate
detector at `src/aelfrice/directive_detector.py` against the labeled
corpus.

### `bfs_relates_to` ship gate (#383)

Track A sub-issue of #382. The `RELATES_TO` edge type is wired schema-side
(`models.EDGE_RELATES_TO`, `bfs_multihop.BFS_EDGE_WEIGHTS[RELATES_TO] = 0.30`,
`triple_extractor` regex emit) but does not ship until it demonstrates a
**≥+5pp BFS multi-hop hit@k uplift on the fixture** vs. the same fixture
run with `BFS_EDGE_WEIGHTS[RELATES_TO]` zeroed. Per #382 Decision A2 the
universal +5pp bar replaces the umbrella's proposed +3pp floor; per A3 there
is no audit-only escape hatch.

Per-row shape:

- `beliefs` — list of `{"id": str, "text": str}` (each id stable within the
  row; harness wires them into a transient `MemoryStore`).
- `edges` — list of `{"src": str, "dst": str, "type": str, "weight": float}`.
  Edge `type` must be one of the live `EDGE_TYPES`; rows can mix edge types
  to test that `RELATES_TO` paths add hits beyond what stronger edges already
  reach.
- `seed_ids` — non-empty list of belief ids the BFS expands from.
- `expected_hit_ids` — non-empty list of belief ids that should be reached
  in the top-k expansion.
- `k` — integer ≥ 1; the rank cutoff used to compute hit@k.

Aggregation: hit-rate is `Σ |reached(row) ∩ expected_hit_ids(row)| /
Σ |expected_hit_ids(row)|` across all rows; uplift is the difference
between the full-weights run and the `RELATES_TO=0` run. Threshold ≥0.05.

Public-tree fixtures may live here (synthetic-only); real-traffic fixtures
stay lab-side per directory-of-origin rules. The bench-gate test at
`tests/bench_gate/test_bfs_multihop_relates_to.py` skips cleanly when the
module dir is empty or has fewer rows than the floor below.

### `implements_edge` ship gate (#385)

Track A sub-issue of #382. The `IMPLEMENTS` edge type is wired schema-side
(`models.EDGE_IMPLEMENTS`, `bfs_multihop.BFS_EDGE_WEIGHTS[IMPLEMENTS] = 0.65`,
`triple_extractor` regex emit) but does not ship until it demonstrates a
**≥+5pp BFS multi-hop hit@k uplift on the fixture** vs. the same fixture
run with `BFS_EDGE_WEIGHTS[IMPLEMENTS]` zeroed. Per #382 Decision A2 the
universal +5pp bar applies.

Per-row shape is identical to `bfs_relates_to`:

- `beliefs` — list of `{"id": str, "text": str}`.
- `edges` — list of `{"src": str, "dst": str, "type": str, "weight": float}`.
  Edge `type` must be one of the live `EDGE_TYPES`; rows should include
  `IMPLEMENTS` edges to test that the edge adds hits beyond what stronger
  edges already reach.
- `seed_ids` — non-empty list of belief ids the BFS expands from.
- `expected_hit_ids` — non-empty list of belief ids that should be reached
  in the top-k expansion.
- `k` — integer ≥ 1; the rank cutoff used to compute hit@k.

Aggregation: hit-rate is `Σ |reached(row) ∩ expected_hit_ids(row)| /
Σ |expected_hit_ids(row)|` across all rows; uplift is the difference
between the full-weights run and the `IMPLEMENTS=0` run. Threshold ≥0.05.

Public-tree fixtures may live here (synthetic-only); real-traffic fixtures
stay lab-side per directory-of-origin rules. The bench-gate test at
`tests/bench_gate/test_bfs_multihop_implements.py` skips cleanly when the
module dir is empty or has fewer rows than the floor below.

### `derived_from_edge` ship gate (#388)

Retroactive bench gate per #382 ratification. `DERIVED_FROM` was added
pre-bench-gate (shipped at v1.2.0 as part of the ingest enrichment wave)
and must now demonstrate the same **≥+5pp BFS multi-hop hit@k uplift on
the fixture** vs. the same fixture run with `BFS_EDGE_WEIGHTS[DERIVED_FROM]`
zeroed. Per #382 Decision A2, the universal +5pp bar applies retroactively;
per A3 there is no audit-only escape hatch.

`DERIVED_FROM` is wired schema-side (`models.EDGE_DERIVED_FROM`,
`bfs_multihop.BFS_EDGE_WEIGHTS[DERIVED_FROM] = 0.70`,
`triple_extractor` regex emit via "is derived from" / "is based on" /
"extends" phrases) and remains in place while the gate is pending.

Per-row shape is identical to `bfs_relates_to` and `implements_edge`:

- `beliefs` — list of `{"id": str, "text": str}`.
- `edges` — list of `{"src": str, "dst": str, "type": str, "weight": float}`.
  Edge `type` must be one of the live `EDGE_TYPES`; rows should include
  `DERIVED_FROM` edges to test that the edge adds hits beyond what stronger
  edges already reach.
- `seed_ids` — non-empty list of belief ids the BFS expands from.
- `expected_hit_ids` — non-empty list of belief ids that should be reached
  in the top-k expansion.
- `k` — integer ≥ 1; the rank cutoff used to compute hit@k.

Aggregation: hit-rate is `Σ |reached(row) ∩ expected_hit_ids(row)| /
Σ |expected_hit_ids(row)|` across all rows; uplift is the difference
between the full-weights run and the `DERIVED_FROM=0` run. Threshold ≥0.05.

Public-tree fixtures may live here (synthetic-only); real-traffic fixtures
stay lab-side per directory-of-origin rules. The bench-gate test at
`tests/bench_gate/test_bfs_multihop_derived_from.py` skips cleanly when the
module dir is empty or has fewer rows than the floor below.

### `temporal_next_edge` ship gate (#386)

Track A sub-issue of #382. The `TEMPORAL_NEXT` edge type is wired schema-side
(`models.EDGE_TEMPORAL_NEXT`, `bfs_multihop.BFS_EDGE_WEIGHTS[TEMPORAL_NEXT] = 0.25`,
`triple_extractor` regex emit) but does not ship until it demonstrates a
**≥+5pp BFS multi-hop hit@k uplift on the fixture** vs. the same fixture
run with `BFS_EDGE_WEIGHTS[TEMPORAL_NEXT]` zeroed. Per #382 Decision A2 the
universal +5pp bar applies (same as all Track A edges).

Per-row shape is identical to `bfs_relates_to`. Aggregation: same formula.
Threshold ≥0.05.

Public-tree fixtures may live here (synthetic-only); real-traffic fixtures
stay lab-side per directory-of-origin rules. The bench-gate test at
`tests/bench_gate/test_bfs_multihop_temporal_next.py` skips cleanly when the
module dir is empty or has fewer rows than the floor below.

### `tests_edge` ship gate (#384)

Track A sub-issue of #382. The `TESTS` edge type is wired schema-side
(`models.EDGE_TESTS`, `bfs_multihop.BFS_EDGE_WEIGHTS[TESTS] = 0.55`,
`triple_extractor` regex emit) but does not ship until it demonstrates a
**≥+5pp BFS multi-hop hit@k uplift on the fixture** vs. the same fixture
run with `BFS_EDGE_WEIGHTS[TESTS]` zeroed. Per #382 Decision A2 the
universal +5pp bar applies (same as all Track A edges).

Per-row shape is identical to `bfs_relates_to`. Aggregation: same formula.
Threshold ≥0.05.

Public-tree fixtures may live here (synthetic-only); real-traffic fixtures
stay lab-side per directory-of-origin rules. The bench-gate test at
`tests/bench_gate/test_bfs_multihop_tests.py` skips cleanly when the
module dir is empty or has fewer rows than the floor below.

## v0.1 acceptance (per #307)

- ≥ 50 non-seed entries per module file (300 total).
- Real examples only — no synthetic generation in v0.1. Synthetic scaffolding
  is permitted in v0.2 after #228 settles consolidation strategy.
- **Carve-outs (synthetic-allowed at v0.1):** `directive_detection/` per #374
  H1 re-entry rules, and `contradiction/` per #201 — the relationship
  detector is a regex/heuristic over modality + quantifier axes where
  controlled synthetic distribution is the cleanest way to exercise each
  axis. Real examples are still preferred and may be mixed in; synthetic
  rows must declare `provenance: "synthetic-vN.M"` so future re-labellers
  can distinguish.
- Schema validation runs in CI (see `tests/test_corpus_schema.py`).

## Validation

`tests/test_corpus_schema.py` enforces:

1. Every `*.jsonl` file under `tests/corpus/v2_0/<module>/` parses line-by-line.
2. Every row has the common envelope fields, non-empty `provenance` and
   `labeller_note`, and a module-allowed `label` value.
3. Every row has the module-specific extra fields with the right shape.
4. `id` values are unique per module.

The test does **not** enforce the ≥50 threshold yet — that flips on once v0.1
labelling lands.
